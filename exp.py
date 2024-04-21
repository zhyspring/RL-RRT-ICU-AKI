#Experiment
import os
import torch
import numpy as np
import os
import pyprind



class DQNExperiment(object):
    def __init__(self, data_loader_train, data_loader_validation, q_network, ps, ns, folder_location, folder_name, saving_period, rng, resume):
        self.rng = rng
        self.data_loader_train = data_loader_train
        self.data_loader_validation = data_loader_validation
        self.q_network = q_network
        self.ps = ps  # num pos samples replaced in each minibatch
        self.ns = ns  # num neg samples replaced in each minibatch
        self.batch_num = 0
        self.saving_period = saving_period  # after each `saving_period` epochs, the results so far will be saved.
        self.resume = resume 
        storage_path = os.path.join(os.path.abspath(folder_location), folder_name)
        self.storage_rl = os.path.join(storage_path, 'rl')
        self.checkpoint_folder = os.path.join(storage_path, 'rl' + '_checkpoints')
        if not os.path.exists(self.storage_rl):
            os.mkdir(self.storage_rl)
        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        print(f'self.checkpoint_folder:{self.checkpoint_folder}')
    def do_epochs(self, number):
        '''
        Each epoch is one iteration thorugh the entire dataset.
        '''
        if self.resume:
            # NOTE: @mehdi: not implemented yet -- now it *overwrites* if project exists
            raise NotImplementedError
            # checkpoint = torch.load(self.checkpoint_file)
            # self.q_network.resume(checkpoint['rl_network_state_dict'], checkpoint['rl_target_network_state_dict'], checkpoint['rl_optimizer_state_dict'])
            # self.curr_epoch = checkpoint['epoch'] + 1
            # self.all_epoch_steps = checkpoint['epoch_steps']
            # self.all_epoch_validation_steps = checkpoint['epoch_validation_steps']
            # self.all_epoch_loss = checkpoint['loss']
            # self.all_epoch_validation_loss = checkpoint['validation_loss']
            # print('Starting from epoch: {0} and continuing upto epoch {1}'.format(self.curr_epoch, number))
        else:
            self.curr_epoch = 0
            self.all_epoch_steps = []
            self.all_epoch_validation_steps = []
            self.all_epoch_loss = []
            self.all_epoch_validation_loss = []
        self.data_loader_train.reset(shuffle=True, pos_samples_in_minibatch=self.ps, neg_samples_in_minibatch=self.ns)
        self.data_loader_validation.reset(shuffle=False, pos_samples_in_minibatch=0, neg_samples_in_minibatch=0)
        for epoch in range(self.curr_epoch, number):
            print()
            print('>>>>> Experiment ' + ' Epoch ' + str(epoch + 1) + '/' + str(number))
            # Learn here
            epoch_done = False
            epoch_steps = 0
            epoch_loss = 0
            print('Minibatch learning within epoch')
            bar = pyprind.ProgBar(self.data_loader_train.num_minibatches_epoch)
            while not epoch_done:
                bar.update()
                s, actions, rewards, next_s, terminals, epoch_done = self.data_loader_train.get_next_minibatch()
                epoch_steps += len(s)
                loss = self.q_network.learn(s, actions, rewards, next_s, terminals)
                epoch_loss += loss
            self.data_loader_train.reset(shuffle=True, pos_samples_in_minibatch=self.ps, neg_samples_in_minibatch=self.ns)
            self.data_loader_validation.reset(shuffle=False, pos_samples_in_minibatch=0, neg_samples_in_minibatch=0)
            self.all_epoch_loss.append(epoch_loss/epoch_steps)
            self.all_epoch_steps.append(epoch_steps)

            if (epoch + 1)% self.saving_period == 0:
                self._do_eval()
                try:
                    torch.save({
                        'epoch': epoch,
                        'rl_network_state_dict': self.q_network.network.state_dict(),
                        # 'rl_target_network_state_dict': self.q_network.target_network.state_dict(),
                        # 'rl_optimizer_state_dict': self.q_network.optimizer.state_dict(),
                        'loss': self.all_epoch_loss,
                        'validation_loss': self.all_epoch_validation_loss,
                        'epoch_steps': self.all_epoch_steps,
                        'epoch_validation_steps': self.all_epoch_validation_steps,
                    }, os.path.join(self.checkpoint_folder, 'checkpoint' + str(epoch) +'.pt'))
                    np.save(os.path.join(self.storage_rl, 'q_losses.npy'), np.array(self.all_epoch_loss))
                    print(f'save rl model at {self.storage_rl}')
                except:
                    print(">>> Cannot save files. On Windows: the files might be open.")
        
    def _do_eval(self):
        epoch_val_steps = 0
        epoch_val_loss = 0
        epoch_done = False
        bar = pyprind.ProgBar(self.data_loader_validation.num_minibatches_epoch)
        while not epoch_done:
            bar.update()
            s, actions, rewards, next_s, terminals, epoch_done = self.data_loader_validation.get_next_minibatch()
            epoch_val_steps += len(s)
            loss = self.q_network.get_loss(s, actions, rewards, next_s, terminals)
            epoch_val_loss += loss
        self.all_epoch_validation_loss.append(epoch_val_loss / epoch_val_steps)
        self.all_epoch_validation_steps.append(epoch_val_steps)
        try:
            np.save(os.path.join(self.storage_rl, 'q_validation_losses.npy'), np.array(self.all_epoch_validation_loss))
        except:
            pass
