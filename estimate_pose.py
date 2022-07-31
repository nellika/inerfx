import os
from pickletools import optimize
import torch
import imageio
import numpy as np
import cv2
from datetime import datetime
from utils import config_parser, load_synth, find_POI, img2mse, rot_phi, to8b
from nerf_helpers import load_nerf
from render_helpers import render, get_rays
from inerf_helpers import camera_transf

def get_batch(batch_size, pool):
    if pool.shape[0] > batch_size: n = pool.shape[0]-1
    rand_inds = np.random.choice(n, size=batch_size, replace=False)
    return pool[rand_inds]

def get_pool(obs_img, H, W, kernel_size, dil_iter, sampling_strategy, feat_det):

    POI = find_POI(obs_img, feat_det)  # xy pixel coordinates of points of interest (N x 2)
    print('poi >>',len(POI))
    obs_img = (np.array(obs_img) / 255.).astype(np.float32)

    # create meshgrid from the observed image
    coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1),
                        dtype=int)

    # create sampling mask for interest region sampling strategy
    interest_regions = np.zeros((H, W, ), dtype=np.uint8)
    interest_regions[POI[:,1], POI[:,0]] = 1
    I = dil_iter
    interest_regions = cv2.dilate(interest_regions, np.ones((kernel_size, kernel_size), np.uint8), iterations=I)
    interest_regions = np.array(interest_regions, dtype=bool)
    interest_regions = coords[interest_regions]

    # not_POI contains all points except of POI
    coords = coords.reshape(H * W, 2)

    if sampling_strategy == 'random': pool = coords
    elif sampling_strategy == 'interest_points': pool = POI
    else: pool = interest_regions
    print(len(coords), len(POI), len(interest_regions))

    return pool

def get_sampled_pixels(obs_img, batch, device, H, W, focal, pose, chunk, render_kwargs):
    target_s = obs_img[batch[:, 1], batch[:, 0]]
    target_s = torch.Tensor(target_s).to(device)

    rays_o, rays_d = get_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3)
    rays_o = rays_o[batch[:, 1], batch[:, 0]]  # (N_rand, 3)
    rays_d = rays_d[batch[:, 1], batch[:, 0]]
    batch_rays = torch.stack([rays_o, rays_d], 0)

    rgb, _, _, _ = render(H, W, focal, chunk=chunk, rays=batch_rays, retraw=True,
                                    **render_kwargs)
    return target_s, rgb

def optimize_params(optim_params, epochs, phi_ref, psi_ref, theta_ref, translation_ref,prefix):
    trans_errs = []
    rot_errs = []
    losses = []

    imgs = []

    for k in range(epochs):

        batch = get_batch(optim_params['batch_size'], optim_params['pool'])


        pose = optim_params['cam_transf'](optim_params['start_pose'])
        target_s, rgb = get_sampled_pixels(optim_params['obs_img'], batch, optim_params['device'], optim_params['H'], optim_params['W'],
                            optim_params['focal'], pose, optim_params['chunk'], optim_params['render_kwargs'])

        optim_params['optimizer'].zero_grad()
        loss = img2mse(rgb, target_s)
        loss.backward(retain_graph=optim_params['retain'])
        optim_params['optimizer'].step()
        new_lrate = optim_params['lrate'] * (0.8 ** ((k + 1) / 100))

        for param_group in optim_params['optimizer'].param_groups:
            param_group['lr'] = new_lrate

        if (k + 1) % 20 == 0 or k == 0:
            with torch.no_grad():
                pose_dummy = pose.cpu().detach().numpy()

                # Rendered pose
                phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
                theta = np.arctan2(-pose_dummy[2, 0], np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2)) * 180 / np.pi
                psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
                translation = np.sqrt(pose_dummy[0,3]**2 + pose_dummy[1,3]**2 + pose_dummy[2,3]**2)

                # Observed vs. rendered pose
                phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                rot_error = phi_error + theta_error + psi_error

                # Logging
                translation_error = abs(translation_ref - translation)
                trans_errs.append(translation_error)
                rot_errs.append(rot_error)
                losses.append(loss.item())

            if optim_params['overlay']:
                with torch.no_grad():
                    rgb, disp, acc, _ = render(optim_params['H'], optim_params['W'], optim_params['focal'], chunk=optim_params['chunk'], c2w=pose[:3, :4], **optim_params['render_kwargs'])
                    rgb = rgb.cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(optim_params['obs_img'])
                    filename = os.path.join(optim_params['testsavedir'], prefix+str(k)+'.png')
                    dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)

    return trans_errs, rot_errs, losses, loss, pose, optim_params['optimizer'], optim_params['cam_transf'], imgs

def run(device):

    # Parameters
    parser = config_parser()
    args = parser.parse_args()
    experiment = args.experiment
    output_dir = args.output_dir
    model_name = args.model_name
    obj_name = args.obj_name
    obs_img_num = args.obs_img_num
    batch_size = args.batch_size
    kernel_size = args.kernel_size
    dil_iter = args.dil_iter
    sampling_strategy = args.sampling_strategy
    feat_det = args.feat_det
    delta_phi, delta_theta, delta_psi = args.delta_phi, args.delta_theta, args.delta_psi
    delta_tx, delta_ty, delta_tz = args.delta_tx, args.delta_ty, args.delta_tz
    logging = args.logging
    overlay = args.overlay
    data_dir = args.data_dir
    white_bkgd = args.white_bkgd
    half_res = args.half_res

    lrate = 0.01

    # Load and pre-process the observed image
    # obs_img - rgb image with elements in range 0...255
    obs_img, hwf, start_pose, obs_img_pose = load_synth(data_dir, obj_name, obs_img_num,
                                            half_res, white_bkgd, delta_phi, delta_theta, delta_psi,
                                            delta_tx, delta_ty, delta_tz)
    H, W, focal = hwf
    near, far = 2., 6.
    # print(obs_img, H, W, kernel_size, dil_iter, sampling_strategy, feat_det)
    # find points of interest of the observed image
    # POI = find_POI(obs_img, feat_det)  # xy pixel coordinates of points of interest (N x 2)
    pool_A = get_pool(obs_img, H, W, kernel_size, dil_iter, sampling_strategy, 'FAST')
    pool_B = get_pool(obs_img, H, W, kernel_size, dil_iter, sampling_strategy, 'BRIEF')
    obs_img = (np.array(obs_img) / 255.).astype(np.float32)

    # Load NeRF Model
    render_kwargs = load_nerf(args, device)
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs.update(bds_dict)

    # Create pose transformation model
    start_pose = torch.Tensor(start_pose).to(device)
    cam_transf_A = camera_transf().to(device)
    optimizer_A = torch.optim.Adam(params=cam_transf_A.parameters(), lr=lrate, betas=(0.9, 0.999))

    cam_transf_B = camera_transf().to(device)
    optimizer_B = torch.optim.Adam(params=cam_transf_B.parameters(), lr=lrate, betas=(0.9, 0.999))

    # Calculate angles and translation of the observed image's pose
    phi_ref = np.arctan2(obs_img_pose[1,0], obs_img_pose[0,0])*180/np.pi
    theta_ref = np.arctan2(-obs_img_pose[2, 0], np.sqrt(obs_img_pose[2, 1]**2 + obs_img_pose[2, 2]**2))*180/np.pi
    psi_ref = np.arctan2(obs_img_pose[2, 1], obs_img_pose[2, 2])*180/np.pi
    translation_ref = np.sqrt(obs_img_pose[0,3]**2 + obs_img_pose[1,3]**2 + obs_img_pose[2,3]**2)


    imgs = []
    imgs_A = []
    imgs_B = []
    imgs_fin = []

    if logging==1:
        start = datetime.now()
        file_name = f'logs/{model_name}_{sampling_strategy}_{experiment}.txt'
        log_file = open(file_name, "a")

        reference = f'{start}\n\nreference phi, theta, psi, translation: {phi_ref}, {theta_ref}, {psi_ref}, {obs_img_pose[0,3]}, {obs_img_pose[1,3]}, {obs_img_pose[2,3]}'
        log_file.write(reference+'\n\n'+'phi,theta,psi,rot_x,rot_y,rot_z,phi_err,theta_err,psi_err\n')

    testsavedir = os.path.join(output_dir, experiment)
    os.makedirs(testsavedir, exist_ok=True)

    first_round_epochs = 150

    optim_params_A = {'overlay':overlay,
                    'batch_size':batch_size,
                    'pool':pool_A,
                    'optimizer':optimizer_A,
                    'retain':True,
                    'last_loss': 0,
                    'lrate':lrate,
                    'cam_transf':cam_transf_A,
                    'start_pose': start_pose,
                    'obs_img':obs_img,
                    'device':device,
                    'H':H,
                    'W':W,
                    'focal':focal,
                    'chunk':args.chunk,
                    'testsavedir': testsavedir,
                    'render_kwargs':render_kwargs}
    trans_errs, rot_errs, losses, loss_A, pose_A, opt_state_A, cam_transf_A, imgs_A = optimize_params(optim_params_A, first_round_epochs, phi_ref, psi_ref, theta_ref, translation_ref,'A_')

    torch.save({
            'optimizer_state_dict': opt_state_A.state_dict(),
            'model_state_dict': cam_transf_A.state_dict(),
            'loss': loss_A,
            }, 'tmp_ckpts/temp_A.pt')
    
    if logging==1:
        errors_log = f'\nErrors:\nrotation,{rot_errs}\ntranslation,{trans_errs}\nloss,\n{losses}\n\n'
        log_file.write(errors_log)
        log_file.write('\nnext optimizer\n')

    optim_params_B = {'overlay':overlay,
                    'batch_size':batch_size,
                    'pool':pool_B,
                    'optimizer':optimizer_B,
                    'retain':True,
                    'last_loss': 0,
                    'lrate':lrate,
                    'cam_transf':cam_transf_B,
                    'start_pose': start_pose,
                    'obs_img':obs_img,
                    'device':device,
                    'H':H,
                    'W':W,
                    'focal':focal,
                    'chunk':args.chunk,
                    'testsavedir': testsavedir,
                    'render_kwargs':render_kwargs}

    trans_errs, rot_errs, losses, loss_B, pose_B, opt_state_B, cam_transf_B, imgs_B = optimize_params(optim_params_B, first_round_epochs, phi_ref, psi_ref, theta_ref, translation_ref,'B_')
    
    torch.save({
            'optimizer_state_dict': opt_state_B.state_dict(),
            'model_state_dict': cam_transf_B.state_dict(),
            'loss': loss_B,
            }, 'tmp_ckpts/temp_B.pt')
    

    if logging==1:
        errors_log = f'\nErrors:\nrotation,{rot_errs}\ntranslation,{trans_errs}\nloss,\n{losses}\n\n'
        log_file.write(errors_log)

    if loss_A.item() < loss_B.item():
        checkpoint = torch.load('tmp_ckpts/temp_A.pt')
        lmsg = f'\nchosen feat det: FAST\n'
        log_file.write(lmsg)
        pool=pool_A
        imgs = imgs_A
    else:
        checkpoint = torch.load('tmp_ckpts/temp_B.pt')
        lmsg = f'\nchosen feat det: BRIEF\n'
        log_file.write(lmsg)
        pool=pool_B
        imgs=imgs_B

    cam_transf = camera_transf().to(device)
    cam_transf.load_state_dict(checkpoint['model_state_dict'])
    prev_lr=checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
    optimizer = torch.optim.Adam(params=cam_transf.parameters(), lr=prev_lr, betas=(0.9, 0.999))

    optim_params = {'overlay':overlay,
                    'batch_size':batch_size,
                    'pool':pool,
                    'optimizer':optimizer,
                    'retain':True,
                    'last_loss': 0,
                    'lrate':lrate,
                    'cam_transf':cam_transf,
                    'start_pose': start_pose,
                    'obs_img':obs_img,
                    'device':device,
                    'H':H,
                    'W':W,
                    'focal':focal,
                    'chunk':args.chunk,
                    'testsavedir': testsavedir,
                    'render_kwargs':render_kwargs}

    trans_errs, rot_errs, losses, loss, _, _, _, imgs_fin = optimize_params(optim_params, 250, phi_ref, psi_ref, theta_ref, translation_ref, 'Fin_')

    print(f'\nfinal loss >> {loss}\n')

    if logging == 1:
        end = datetime.now()
        errors_log = f'\nErrors:\nrotation,{rot_errs}\ntranslation,{trans_errs}\nloss,\n{losses}\n\n{end}'
        log_file.write(errors_log)
        log_file.close()


    if overlay is True:
        frames = imgs + imgs_fin
        imageio.mimwrite(os.path.join(testsavedir, 'video.gif'), frames, fps=8) #quality = 8 for mp4 format

