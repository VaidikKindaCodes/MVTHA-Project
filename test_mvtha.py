import torch

from model.mvtha import Model


def print_stage_shape(name):
    def hook(_module, _inputs, output):
        print('{} output shape: {}'.format(name, output.shape))
    return hook


def main():
    model = Model(num_class=60, num_point=25, num_person=2, in_channels=3)
    x = torch.randn(2, 3, 64, 25, 2)
    print('Input shape: {}'.format(x.shape))

    stages = (
        ('VTM', model.view_transformation),
        ('MVT', model.multi_view_transformer),
        ('APAM', model.apam),
        ('HMSAM', model.hmsam),
        ('MS-TC', model.multi_scale_temporal_conv),
    )
    handles = [module.register_forward_hook(print_stage_shape(name))
               for name, module in stages]
    try:
        output = model(x)
    finally:
        for handle in handles:
            handle.remove()

    print('Final output shape: {}'.format(output.shape))
    assert output.shape == torch.Size([2, 60])

    loss = output.mean()
    loss.backward()
    print('Backward pass: successful')


if __name__ == '__main__':
    main()
