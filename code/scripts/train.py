import os
from pyhocon import ConfigFactory
import torch

import sys
sys.path.append('./')

import utils.general as utils
import utils.plots as plt

import wandb
from functools import partial
from model.point_avatar_model import PointAvatar
from model.loss import Loss
print = partial(print, flush=True)


class TrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        # max_batch:8
        self.max_batch = self.conf.get_int('train.max_batch', default='8')
        # max_points_training: 409600, num_batch * num_points
        self.batch_size = min(int(self.conf.get_int('train.max_points_training') / self.conf.get_int('model.point_cloud.n_init_points')), self.max_batch)
        # n
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        # set environment variables for wandb
        # init wandb
        os.environ['WANDB_DIR'] = os.path.join(self.exps_folder_name)
        wandb.init(project=kwargs['wandb_workspace'], name=self.subject + '_' + self.methodname, config=self.conf, tags=kwargs['wandb_tags'])

        self.optimize_inputs = self.optimize_expression or self.optimize_pose
        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['is_continue']:
            if kwargs['load_path'] != '':
                load_path = kwargs['load_path']
            else:
                load_path = self.train_dir
            if os.path.exists(os.path.join(load_path)):
                is_continue = True
            else:
                is_continue = False
        else:
            is_continue = False

        utils.mkdir_ifnotexists(self.train_dir)
        utils.mkdir_ifnotexists(self.eval_dir)

        # region create checkpoints dirs
        self.checkpoints_path = os.path.join(self.train_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))
        # endregion

        if self.optimize_inputs:
            self.optimizer_inputs_subdir = "OptimizerInputs"
            self.input_params_subdir = "InputParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.input_params_subdir))

        # copy config file
        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.train_dir, 'runconf.conf')))
        # backup code files
        self.file_backup() 

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.use_background = self.conf.get_bool('dataset.use_background', default=False)
        self.sub_view_train = self.conf.get_list('dataset.train.sub_view', default=[]) # change
        self.sub_view_plot = self.conf.get_list('dataset.test.sub_view', default=[]) # change
        
        self.n_views = len(self.sub_view_train)
        print('sub_view_train: ', self.sub_view_train)
        print('sub_view_plot: ', self.sub_view_plot)
        # instantiate a dataset object
        # json_name: 'flame_params.json'
        # change the dataset class from single view to multi views
        if 'sub_view' in self.conf.get_config('dataset.train'):
            del self.conf.get_config('dataset.train')['sub_view']
            print('sub_view_train is deleted from config file\n')
        if 'sub_view' in self.conf.get_config('dataset.test'):
            del self.conf.get_config('dataset.test')['sub_view']
            print('sub_view_plot is deleted from config file\n')
        # self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                        #   subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                        #   json_name=self.conf.get_string('dataset.json_name'),
                                                                                        #   use_mean_expression=self.conf.get_bool('dataset.use_mean_expression', default=False),
                                                                                        #   use_var_expression=self.conf.get_bool('dataset.use_var_expression', default=False),
                                                                                        #   use_background=self.use_background,
                                                                                        #   is_eval=False,
                                                                                        #   **self.conf.get_config('dataset.train'))

        # self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                        #  subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                        #  json_name=self.conf.get_string('dataset.json_name'),
                                                                                        #  use_background=self.use_background,
                                                                                        #  is_eval=True,
                                                                                        #  **self.conf.get_config('dataset.test'))

        self.train_datasets, self.plot_datasets = [], []
        for view in self.sub_view_train:
            self.train_datasets.append(utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         use_mean_expression=self.conf.get_bool('dataset.use_mean_expression', default=False),
                                                                                         use_var_expression=self.conf.get_bool('dataset.use_var_expression', default=False),
                                                                                         use_background=self.use_background,
                                                                                         is_eval=False,
                                                                                         sub_view=view,
                                                                                         **self.conf.get_config('dataset.train')))
        for view in self.sub_view_plot:
            self.plot_datasets.append(utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         use_background=self.use_background,
                                                                                         is_eval=True,
                                                                                         sub_view=view,
                                                                                         **self.conf.get_config('dataset.test')))
        self.train_dataset = self.train_datasets[0]
        self.plot_dataset = self.plot_datasets[0]
        # print train_datasets length
        print('train_datasets length: ', len(self.train_datasets))
        print('Finish loading data ...')
        
        # change: shape_params, mean_expression use mean of all views instead of single view

        total_shape_params = torch.zeros_like(self.train_datasets[0].shape_params)
        total_expression = torch.zeros_like(self.train_datasets[0].mean_expression)
        total_var_expression = torch.zeros_like(self.train_datasets[0].var_expression)
        
        for dataset in self.train_datasets:
            total_shape_params += dataset.shape_params
            total_expression += dataset.mean_expression
            total_var_expression += dataset.var_expression
        
        self.mean_shape_params = total_shape_params / self.n_views
        self.mean_expression = total_expression / self.n_views
        self.mean_expression_var = total_var_expression / self.n_views
        
        # todo: change shape_params, mean_expression
        self.model = PointAvatar(conf=self.conf.get_config('model'),
                                shape_params=self.mean_shape_params, # change
                                img_res=self.train_dataset.img_res,
                                canonical_expression=self.mean_expression, # change
                                canonical_pose=self.conf.get_float('dataset.canonical_pose', default=0.2),
                                use_background=self.use_background)
        
        self._init_dataloader()
        if torch.cuda.is_available():
            self.model.cuda()

        # change
        self.loss = Loss(**self.conf.get_config('loss'), var_expression=self.mean_expression_var)

        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam([
            {'params': list(self.model.parameters())},
        ], lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)
        self.upsample_freq = self.conf.get_int('train.upsample_freq', default=5)
        # settings for input parameter optimization
        # todo: check if need to optimize pose
        if self.optimize_inputs:
            num_training_frames = len(self.train_dataset)
            param = []
                
            if self.optimize_expression:
                # change:将每个dataset中的expressions相加取平均
                total_expressions = []
                for dataset in self.train_datasets:
                    total_expressions.append(dataset.data["expressions"])
                mean_expressions = torch.mean(torch.stack(total_expressions), dim=0)
                
                init_expression = torch.cat((mean_expressions, torch.randn(self.train_dataset.data["expressions"].shape[0], max(self.model.deformer_network.num_exp - 50, 0)).float()), dim=1)
                self.expression = torch.nn.Embedding(num_training_frames, self.model.deformer_network.num_exp, _weight=init_expression, sparse=True).cuda()
                param += list(self.expression.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(num_training_frames, 15, _weight=self.train_dataset.data["flame_pose"], sparse=True).cuda()
                self.camera_pose = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.data["world_mats"][:, :3, 3], sparse=True).cuda()
                param += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())
            self.optimizer_cam = torch.optim.SparseAdam(param, self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0
        # region load checkpoints
        if is_continue:
            old_checkpnts_dir = os.path.join(load_path, 'checkpoints')
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.start_epoch = saved_model_state['epoch']
            n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]

            batch_size = min(int(self.conf.get_int('train.max_points_training') / n_points), self.max_batch)
            if self.batch_size != batch_size:
                self.batch_size = batch_size
                self._init_dataloader()
            self.model.pc.init(n_points)
            self.model.pc = self.model.pc.cuda()

            self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)

            self.model.raster_settings.radius = saved_model_state['radius']

            self.optimizer = torch.optim.Adam([
                {'params': list(self.model.parameters())},
            ], lr=self.lr)

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.optimize_inputs:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_inputs_subdir, str(kwargs['checkpoint']) + ".pth"))
                try:
                    self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])
                except:
                    print("input and camera optimizer parameter group doesn't match")
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                try:
                    if self.optimize_expression:
                        self.expression.load_state_dict(data["expression_state_dict"])
                    if self.optimize_pose:
                        self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                        self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
                except:
                    print("expression or pose parameter group doesn't match")
        # endregion
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=4,
                                                            )
        self.n_batches = len(self.train_dataloader)
        self.img_res = self.plot_dataset.img_res
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.save_freq = self.conf.get_int('train.save_freq', default=1)

        self.GT_lbs_milestones = self.conf.get_list('train.GT_lbs_milestones', default=[])
        self.GT_lbs_factor = self.conf.get_float('train.GT_lbs_factor', default=0.5)
        for acc in self.GT_lbs_milestones:
            if self.start_epoch > acc:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
        # if len(self.GT_lbs_milestones) > 0 and self.start_epoch >= self.GT_lbs_milestones[-1]:
        #    self.loss.lbs_weight = 0.

    def _init_dataloader(self):
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=4,
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=min(self.batch_size * 2, 10),
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn,
                                                           num_workers=4,
                                                           )
        # change from single view to multi views
        self.train_dataloaders, self.plot_dataloaders = [], []
        for dataset in self.train_datasets:
            self.train_dataloaders.append(torch.utils.data.DataLoader(dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=dataset.collate_fn,
                                                            num_workers=4,
                                                            ))
        for dataset in self.plot_datasets:
            self.plot_dataloaders.append(torch.utils.data.DataLoader(dataset,
                                                            batch_size=min(self.batch_size * 2, 10),
                                                            shuffle=False,
                                                            collate_fn=dataset.collate_fn,
                                                            num_workers=4,
                                                            ))
        self.n_batches = len(self.train_dataloader)
    def save_checkpoints(self, epoch, only_latest=False):
        if not only_latest:
            torch.save(
                {"epoch": epoch, "radius": self.model.raster_settings.radius,
                 "model_state_dict": self.model.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
                os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))

        torch.save(
            {"epoch": epoch, "radius": self.model.raster_settings.radius,
                 "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.optimize_inputs:
            if not only_latest:
                torch.save(
                    {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                    os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, "latest.pth"))\

            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            if not only_latest:
                torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, str(epoch) + ".pth"))
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, "latest.pth"))

    def upsample_points(self):
        current_radius = self.model.raster_settings.radius
        points = self.model.pc.points.data
        noise = (torch.rand(*points.shape).cuda() - 0.5) * current_radius
        new_points = noise + points
        self.model.pc.upsample_points(new_points)
        self.model.raster_settings.radius = 0.75 * current_radius
        print("old radius: {}, new radius: {}".format(current_radius, self.model.raster_settings.radius))
        print("old points: {}, new points: {}".format(self.model.pc.points.data.shape[0]/2, self.model.pc.points.data.shape[0]))
        self.optimizer = torch.optim.Adam([
            {'params': list(self.model.parameters())},
        ], lr=self.lr)

    # backup code files
    def file_backup(self):
        from shutil import copyfile
        dir_lis = ['./model', './scripts', './utils', './flame', './datasets']
        os.makedirs(os.path.join(self.train_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.train_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        # copyfile(self.conf_path, os.path.join(self.train_dir, 'recording', 'config.conf'))

    def run(self):
        acc_loss = {} # accumulate loss
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        print('Start training ...')
        # training loop
        for epoch in range(self.start_epoch, self.nepochs + 1): # 0 ~ nepochs
            if epoch in self.GT_lbs_milestones:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
            # if len(self.GT_lbs_milestones) > 0 and epoch >= self.GT_lbs_milestones[-1]:
            #   self.loss.lbs_weight = 0.

            # region save checkpoints
            if epoch % (self.save_freq * 5) == 0 and epoch != self.start_epoch:
                self.save_checkpoints(epoch)
            else:
                if epoch % self.save_freq == 0 and (epoch != self.start_epoch or self.start_epoch == 0):
                    self.save_checkpoints(epoch, only_latest=True)
            # endregion
            if (epoch % self.plot_freq == 0 and epoch < 5) or (epoch % (self.plot_freq * 2) == 0):
                self.model.eval()
                if self.optimize_inputs:
                    if self.optimize_expression:
                        self.expression.eval()
                    if self.optimize_pose:
                        self.flame_pose.eval()
                        self.camera_pose.eval()
                eval_iterator = iter(self.plot_dataloader) 
                start_time.record()
                
                # change
                self.eval_iters = [iter(dataloader) for dataloader in self.plot_dataloaders]
                for batch_index in range(len(self.plot_dataloader)):
                    for i, eval_iterator in enumerate(self.eval_iters):
                        indices, model_input, ground_truth = next(eval_iterator)
                        for k, v in model_input.items():
                            try:
                                model_input[k] = v.cuda()
                            except:
                                model_input[k] = v
                        for k, v in ground_truth.items():
                            try:
                                ground_truth[k] = v.cuda()
                            except:
                                ground_truth[k] = v

                        model_outputs = self.model(model_input)
                        for k, v in model_outputs.items():
                            try:
                                model_outputs[k] = v.detach()
                            except:
                                model_outputs[k] = v
                        plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_'+str(epoch)) for i in range(len(model_input['sub_dir']))]
                        img_names = model_input['img_name'][:, 0].cpu().numpy()
                        print("Plotting images: {}".format(img_names))
                        utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))

                        plt.plot(img_names,
                                model_outputs,
                                ground_truth,
                                plot_dir,
                                epoch,
                                self.img_res,
                                is_eval=False,
                                first=(batch_index==0),
                                )
                        del model_outputs, ground_truth

                end_time.record()
                torch.cuda.synchronize()
                print("Plot time per image: {} ms".format(start_time.elapsed_time(end_time) / len(self.plot_dataset)))
                self.model.train()
                if self.optimize_inputs:
                    if self.optimize_expression:
                        self.expression.train()
                    if self.optimize_pose:
                        self.flame_pose.train()
                        self.camera_pose.train()

            start_time.record()
            
            # Prunning
            if epoch != self.start_epoch and self.model.raster_settings.radius >= 0.006:
                self.model.pc.prune(self.model.visible_points)
                self.optimizer = torch.optim.Adam([
                    {'params': list(self.model.parameters())},
                ], lr=self.lr)
            # Upsampling
            if epoch % self.upsample_freq == 0:
                if epoch != 0 and self.model.pc.points.shape[0] <= self.model.pc.max_points / 2:
                    self.upsample_points()
                    batch_size = min(int(self.conf.get_int('train.max_points_training') / self.model.pc.points.shape[0]), self.max_batch)
                    if batch_size != self.batch_size:
                        self.batch_size = batch_size
                        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                                            batch_size=self.batch_size,
                                                                            shuffle=True,
                                                                            collate_fn=self.train_dataset.collate_fn,
                                                                            num_workers=4,
                                                                            )
                        self.n_batches = len(self.train_dataloader)

            # re-init visible point tensor each epoch
            self.model.visible_points = torch.zeros(self.model.pc.points.shape[0]).bool().cuda()
            self.train_iters = [iter(dataloader) for dataloader in self.train_dataloaders]
            for batch_index in range(len(self.train_dataloader)):
                all_loss = []
                all_loss_output = []
                
                self.optimizer.zero_grad()
                if self.optimize_inputs and epoch > 10:
                    self.optimizer_cam.zero_grad()
                    
                for i, dataloader in enumerate(self.train_iters):
                    indices, model_input, ground_truth = next(dataloader)
                    for k, v in model_input.items():
                        try:
                            model_input[k] = v.cuda()
                        except:
                            model_input[k] = v
                    for k, v in ground_truth.items():
                        try:
                            ground_truth[k] = v.cuda()
                        except:
                            ground_truth[k] = v
                    if self.optimize_inputs:
                        if self.optimize_expression:
                                model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                        if self.optimize_pose:
                            model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                            model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)

                    model_outputs = self.model(model_input)
                        
                    loss_output = self.loss(model_outputs, ground_truth)

                    loss = loss_output['loss']
                        
                    loss.backward()
                    
                    all_loss.append(loss)
                    all_loss_output.append(loss_output)

                self.optimizer.step()
                if self.optimize_inputs and epoch > 10:
                    self.optimizer_cam.step()

                # for k, v in loss_output.items():
                #     loss_output[k] = v.detach().item()
                #     if k not in acc_loss:
                #         acc_loss[k] = [v]
                #     else:
                #         acc_loss[k].append(v)

                # acc_loss['visible_percentage'] = (torch.sum(self.model.visible_points)/self.model.pc.points.shape[0]).unsqueeze(0)
                # if data_index % 50 == 0:
                #     for k, v in acc_loss.items():
                #         acc_loss[k] = sum(v) / len(v)
                #     print_str = '{0} [{1}] ({2}/{3}): '.format(self.methodname, epoch, data_index, self.n_batches)
                #     for k, v in acc_loss.items():
                #         print_str += '{}: {:.3g} '.format(k, v)
                #     print(print_str)
                #     acc_loss['num_points'] = self.model.pc.points.shape[0]
                #     acc_loss['radius'] = self.model.raster_settings.radius

                #     acc_loss['lr'] = self.scheduler.get_last_lr()[0]
                #     wandb.log(acc_loss, step=epoch * len(self.train_dataset) + data_index * self.batch_size)
                #     acc_loss = {}

            self.scheduler.step()
            end_time.record()
            torch.cuda.synchronize()
            # wandb.log({"timing_epoch": start_time.elapsed_time(end_time)}, step=(epoch+1) * len(self.train_dataset))
            print("Epoch time: {} s".format(start_time.elapsed_time(end_time)/1000))
        self.save_checkpoints(self.nepochs + 1)



