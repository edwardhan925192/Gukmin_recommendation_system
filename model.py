import torch
from kernel_layer import KernelLayer
from global_conv import GlobalKernel, GlobalConv

class Model(nn.Module):
    def __init__(self, n_u, n_layers, n_hid, n_dim, lambda_s, lambda_2, gk_size, dot_scale):
        super(Model, self).__init__()
        self.n_layers = n_layers
        self.kernel_layers = nn.ModuleList([KernelLayer(n_hid, n_hid, n_dim, lambda_s, lambda_2) for _ in range(n_layers)])
        self.output_layer = KernelLayer(n_hid, n_u, n_dim, lambda_s, lambda_2, activation=lambda x: x)
        self.global_kernel = GlobalKernel(gk_size, dot_scale)
        self.global_conv = GlobalConv()

    def pretrain(self, train_r, train_m):
        y = train_r
        reg_losses = 0
        for i in range(self.n_layers):
            y, reg_loss = self.kernel_layers[i](y)
            reg_losses += reg_loss
        pred_p, reg_loss = self.output_layer(y)
        reg_losses += reg_loss
        diff = train_m * (train_r - pred_p)
        sqE = torch.norm(diff, p=2)**2
        loss_p = sqE + reg_losses
        return loss_p, pred_p

    def finetune(self, train_r, train_m):
        y = train_r
        reg_losses = 0
        for i in range(self.n_layers):
            y, _ = self.kernel_layers[i](y)
        y_dash, _ = self.output_layer(y)
        gk = self.global_kernel(y_dash)
        y_hat = self.global_conv(train_r, gk)
        for i in range(self.n_layers):
            y_hat, reg_loss = self.kernel_layers[i](y_hat)
            reg_losses += reg_loss
        pred_f, reg_loss = self.output_layer(y_hat)
        reg_losses += reg_loss
        diff = train_m * (train_r - pred_f)
        sqE = torch.norm(diff, p=2)**2
        loss_f = sqE + reg_losses
        return loss_f, pred_f

    def forward(self, train_r, train_m, phase='pretrain'):
        if phase == 'pretrain':
            return self.pretrain(train_r, train_m)
        elif phase == 'finetune':
            return self.finetune(train_r, train_m)
        else:
            raise ValueError("Invalid phase. Choose either 'pretrain' or 'finetune'.")
