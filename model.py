import kernel_layer
from kernel_layer import KernelLayer
from global_conv import GlobalKernel, GlobalConv

class Model(nn.Module):
    def __init__(self, n_layers,n_in, n_hid, n_dim, lambda_s, lambda_2, gk_size, dot_scale):
        super(Model, self).__init__()
        self.n_layers = n_layers
        self.n_hid = n_hid
        self.n_dim = n_dim
        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2
        self.kernel_layers = nn.ModuleList([KernelLayer(n_u, n_hid, n_dim, lambda_s, lambda_2)])

        for _ in range(1, n_layers):
            self.kernel_layers.append(KernelLayer(n_hid, n_hid, n_dim, lambda_s, lambda_2))

        self.output_layer = KernelLayer(n_hid, n_u, n_dim, lambda_s, lambda_2, activation=lambda x: x)
        self.global_kernel = GlobalKernel(gk_size, dot_scale)
        self.global_conv = GlobalConv()

    def pretrain(self, train_r, train_m):
        y = train_r
        reg_losses = 0

        # =========== First reconstruction(pretraining) ========= #
        for i in range(self.n_layers):
            y, reg_loss = self.kernel_layers[i](y)
            reg_losses += reg_loss

        # =========== Decoding (pretraining) ========= #
        pred_p, reg_loss = self.output_layer(y)
        reg_losses += reg_loss

        diff = (train_m - pred_p)
        sqE = torch.norm(diff, p=2)**2
        loss_p = sqE + reg_losses
        return loss_p, pred_p

    def finetune(self, train_r, train_m):
        y = train_r
        reg_losses = 0

        for i in range(self.n_layers):
            y, reg_loss = self.kernel_layers[i](y)
            reg_losses += reg_loss

        y_dash, reg_loss = self.output_layer(y)
        reg_losses += reg_loss

        # ============= R hat ============== #
        gk = self.global_kernel(y_dash)
        y_hat = self.global_conv(train_r, gk)

        print(f'y_hat: {y_hat.shape}')
        y_hat = y_hat.squeeze(0)

        # ============= Reconstructing R hat  ============== #
        for i in range(self.n_layers):
            y_hat, reg_loss = self.kernel_layers[i](y_hat)
            reg_losses += reg_loss


        pred_f, reg_loss = self.output_layer(y_hat)
        reg_losses += reg_loss

        # L2 loss
        diff = (train_m - pred_f)
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
