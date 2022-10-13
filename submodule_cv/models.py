import torchvision.models as models
import torch.nn as nn
import torch
import os
from submodule_cv.deep_models.models import Model
from efficientnet_pytorch.model import MBConvBlock
import numpy as np
import torch


class BaseModel():
    def name(self):
        return 'base_model'

    def __init__(self, config):
        self.config = config
        self.use_weighted_loss = config["use_weighted_loss"]
        self.continue_train = config["continue_train"]
        self.freeze_ = True if "freeze" in config and config["freeze"]!=-1 else False
        self.trainable_layer_num = config["freeze"] if self.freeze_ else None

    def forward(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_errors(self):
        pass

    def save_state(self):
        pass

    def load_state(self):
        pass

class DeepModel(BaseModel):
    def name(self):
        n = [self.deep_model]
        if self.use_pretrained:
            n += ['pretrained_weights']
        if self.use_weighted_loss:
            n += ['weighted_loss']
        if self.use_antialias:
            n += ['antialias']
        return '_'.join(n)

    def __init__(self, config, is_eval=False, class_weight=None, device=None):
        """
        TODO: very messy. Should clean this.
        TODO: is_eval param is not being used. Should use
        TODO: NNs should use 1 output neuron for binary classification instead of 2. Should refactor to use 1 output with last linear layer 1 output then sigmoid
        https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons
        """
        super().__init__(config)
        print("Config File:")
        print(config)
        self.is_eval = is_eval
        self.deep_model = self.config["model"]["base_model"]
        self.class_weight = class_weight if self.use_weighted_loss else None
        self.MixUp = True if 'mix_up' in self.config and self.config['mix_up']['use_mix_up'] else False

        model = Model(config["model"])

        print(model)
        if device is not None:
            self.model = torch.nn.DataParallel(model, device_ids=range(0,len(device))).cuda()
            self.model = model.to(f'cuda:0')
        else:
            if torch.cuda.is_available():
                self.model = model.cuda()
            else:
                self.model = model

        if not self.is_eval:
            if self.use_weighted_loss and self.class_weight is not None:
                weight = torch.Tensor(self.class_weight).cuda()
                self.criterion = torch.nn.CrossEntropyLoss(
                    reduction='mean', weight=weight)
                #self.criterion = torch.nn.BCEWithLogitsLoss(
                #    reduction='mean', weight=torch.from_numpy(self.class_weight).cuda())
            else:
                self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
                #self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        print("Parameters to learn:")
        params = self.find_trainable_parameters()
        params_to_update = []
        # TODO
        # if self.freeze_:
        # else:
        for layer in ["feature_extract", "classifier"]:
            # Check if it is empty
            if params[layer]:
                params_to_update.append({"params": params[layer]})
        assert params_to_update, f"There is no parameter for optimizer!"

        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

        optimizer = getattr(torch.optim, self.config["optimizer"]["type"])
        self.optimizer = optimizer(params_to_update, **self.config["optimizer"]["parameters"])

        if "scheduler" in self.config:
            scheduler = getattr(torch.optim.lr_scheduler, self.config["scheduler"]["type"])
            self.scheduler = scheduler(self.optimizer, **self.config["scheduler"]["parameters"])

        if self.continue_train:
            self.load_state(config["load_deep_model_id"], device=device)

        if self.is_eval:
            self.load_state(config["load_deep_model_id"], device=device)
            self.model = self.model.eval()

    def forward(self, input_data):
        output = self.model.forward(input_data)
        if type(output).__name__ in ['GoogLeNetOutputs', 'InceptionOutputs'] and config["parameters"]["aux_logits"]:
            logits = output.logits
        else:
            logits = output
        probs = torch.softmax(logits, dim=1)
        return logits, probs, output

    def get_loss(self, logits, labels, output=None,
                 labels_mixed=None, lam=None):
        if type(output).__name__ == 'GoogLeNetOutputs':
            loss = self.criterion(logits.type(torch.float), labels.type(torch.long)) + 0.4 * (self.criterion(output.aux_logits1.type(
                torch.float), labels.type(torch.long)) + self.criterion(output.aux_logits2.type(torch.float), labels.type(torch.long)))
        elif type(output).__name__ == 'InceptionOutputs':
            loss = self.criterion(logits.type(torch.float), labels.type(
                torch.long)) + 0.4 * self.criterion(output.aux_logits.type(torch.float), labels.type(torch.long))
        else:
            if self.MixUp and labels_mixed is not None and lam is not None:
                loss = lam * self.criterion(logits.type(
                    torch.float), labels.type(torch.long)) + (1 - lam) * self.criterion(logits.type(
                        torch.float), labels_mixed.type(torch.long))
            else:
                loss = self.criterion(logits.type(
                    torch.float), labels.type(torch.long))
        return loss

    def optimize_parameters(self, logits, labels, output=None,
                            labels_mixed=None, lam=None):
        self.optimizer.zero_grad()
        self.loss = self.get_loss(logits, labels, output, labels_mixed, lam)
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    def get_current_errors(self):
        return self.loss.item()

    def scheduler_step(self):
        self.scheduler.step()

    def get_current_lr(self, grp_idx=0):
        return self.optimizer.param_groups[grp_idx]['lr']

    def load_state(self, save_path, device=None):
        if device:
            state = torch.load(save_path, map_location=device)
        elif not torch.cuda.is_available():
            state = torch.load(save_path, map_location='cpu')
        else:
            state = torch.load(save_path)

        try:
            self.model.load_state_dict(state['state_dict'])
        except RuntimeError:
            pretrained_dict = state['state_dict']
            model_dict = self.model.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.model.load_state_dict(pretrained_dict)

        self.optimizer.load_state_dict(state['optimizer'])

        model_id = state['iter_idx']
        return model_id

    def load_state_old_version(self, save_path, device=None):
        '''
        This function load the model trained on previous singularity_train
        version to the newest one.

        NOTE: it does not load the optimizer. it used only for testing, not
        continuing training!
        NOTE: it does not support model trained on multi-GPUs (TODO)
        '''
        if device:
            state = torch.load(save_path, map_location=device)
        elif not torch.cuda.is_available():
            state = torch.load(save_path, map_location='cpu')
        else:
            state = torch.load(save_path)

        cur_state = self.model.state_dict()
        new_state = {}
        for (old, new) in zip(state['state_dict'].items(), cur_state.items()):
            _, old_pt = old
            new_name, _ = new
            new_state[new_name] = old_pt
        self.model.load_state_dict(new_state)

        model_id = state['iter_idx']
        return model_id

    def load_state_repr(self, save_path, device=None):
        '''
        This function load the model trained on representation learning

        NOTE: it does not load the optimizer. it used only for testing, not
        continuing training!
        NOTE: it does not support model trained on multi-GPUs (TODO)
        '''
        if device:
            state = torch.load(save_path, map_location=device)
        elif not torch.cuda.is_available():
            state = torch.load(save_path, map_location='cpu')
        else:
            state = torch.load(save_path)

        # Approach 1
        # cur_state = self.model.state_dict()
        # new_state = {}
        # for (repr, cur) in zip(state.items(), cur_state.items()):
        #     repr_name, repr_pt = repr
        #     cur_name, cur_pt = cur
        #     if repr_name.startswith('feature_extract.'):
        #         assert repr_name==cur_name, f"{repr_name} is not same as {cur_name}"
        #         new_state[cur_name] = repr_pt
        #
        # self.model.load_state_dict(new_state, strict=False)

        # Approach 1
        self.model.load_state_dict(state, strict=False)

    def save_state(self, save_location, train_instance_name, iter_idx, epoch):
        filename = f'{train_instance_name}.pth'
        save_path = os.path.join(save_location, filename)
        os.makedirs(save_location, exist_ok=True)
        state = {
            'epoch': epoch,
            'iter_idx': iter_idx,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, save_path)

    def freeze_except_bn(self):
        def freeze_all_but_bn(layer):
            if not isinstance(layer, torch.nn.BatchNorm2d) and not isinstance(layer, torch.nn.BatchNorm1d):
                if hasattr(layer, 'weight') and layer.weight is not None:
                    layer.weight.requires_grad_(False)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.requires_grad_(False)
        self.model.apply(freeze_all_but_bn)

    def freeze_all(self):
        def freeze_all_(layer):
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.requires_grad_(False)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.requires_grad_(False)
        self.model.apply(freeze_all_)

    def make_classifier_layer_trainable(self):
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def make_feature_extractor_layer_trainable(self):
        for param in self.model.feature_extract.parameters():
            param.requires_grad = True

    def unfreeze_reverse_layers(self, num_trainable_layers):
        rev_child = reversed(list(self.model.children()))
        def unfreeze_layers(rev_child, num):
            for layer in rev_child:
                if num > 0:
                    if isinstance(layer, nn.Sequential) or \
                    isinstance(layer, models.resnet.BasicBlock) or \
                    isinstance(layer, models.resnet.Bottleneck) or \
                    isinstance(layer, models.squeezenet.Fire) or \
                    isinstance(layer, nn.ModuleList) or \
                    isinstance(layer, MBConvBlock):
                        num = unfreeze_layers(reversed(list(layer.children())), num)
                    else:
                        flag = False
                        if not isinstance(layer, torch.nn.BatchNorm2d) and not isinstance(layer, torch.nn.BatchNorm1d):
                            if hasattr(layer, 'weight') and layer.weight is not None:
                                layer.weight.requires_grad_(True)
                                flag = True
                            if hasattr(layer, 'bias') and layer.bias is not None:
                                layer.bias.requires_grad_(True)
                                flag = True
                            if flag:
                                num -= 1
            return num
        unfreeze_layers(rev_child, num_trainable_layers)

    def find_trainable_parameters(self):
        params_to_update = {"feature_extract": [], "classifier": []}
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                if "feature_extract" in name:
                    params_to_update["feature_extract"].append(param)
                elif "classifier" in name:
                    params_to_update["classifier"].append(param)
                else:
                    raise ValueError(f"{name} should be either _feature_extract_ or _classifier_!")
        return params_to_update

    def update_optimizer_schedular(self, learning_rate=None, use_scheduler=True, epoch=None, batch_per_epoch=None):
        lr = {}
        if learning_rate is None:
            learning_rate = self.config["optimizer"]["parameters"]["lr"]
        if isinstance(learning_rate, float):
            lr["classifier"]      = learning_rate
            lr["feature_extract"] = learning_rate / 10
        elif isinstance(learning_rate, list):
            assert len(learning_rate)==2, f"There should be two numbers in the list"
            lr["classifier"]      = max(learning_rate)
            lr["feature_extract"] = min(learning_rate)

        params    = self.find_trainable_parameters()
        params_lr = []
        max_lr    = []
        log       = ""
        for layer in ["feature_extract", "classifier"]:
            # Check if it is empty
            if params[layer]:
                params_lr.append({"params": params[layer], "lr": lr[layer]})
                max_lr.append(lr[layer])
                log += f"-{layer}: lr = {lr[layer]} "

        assert params_lr, f"There is not parameter for optimizer!"

        optimizer = getattr(torch.optim, self.config["optimizer"]["type"])
        self.optimizer = optimizer(params_lr, **self.config["optimizer"]["parameters"])

        if use_scheduler:
            scheduler_type  = self.config["scheduler"]["type"] if "scheduler" in self.config else "OneCycleLR"
            scheduler_param = self.config["scheduler"]["parameters"] if "scheduler" in self.config else {}
            if scheduler_type=="OneCycleLR":
                scheduler_param["max_lr"] = max_lr
                scheduler_param["epochs"] = epoch
                scheduler_param["steps_per_epoch"] = batch_per_epoch
                if isinstance(learning_rate, float):
                    scheduler_param["pct_start"] = 0.99
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)
            self.scheduler = scheduler(self.optimizer, **scheduler_param)
        # Log
        opt_type = self.config["optimizer"]["type"]
        print(f"\tOptimizer: {opt_type}: {log}")
        if use_scheduler:
            print(f"\tScheduler: {scheduler_type}!")

    def unfreeze(self, num_trainable_layers=-1):
        if num_trainable_layers==-1:
            self.make_classifier_layer_trainable()
            self.make_feature_extractor_layer_trainable()
        else:
            self.unfreeze_reverse_layers(num_trainable_layers)

    def freeze(self, num_trainable_layers=-1):
        self.freeze_except_bn()
        if num_trainable_layers==-1:
            self.make_classifier_layer_trainable()
        else:
            self.unfreeze_reverse_layers(num_trainable_layers)
