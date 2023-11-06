_base_ = [
    '../_base_/models/fccdn_se.py', 
    '../common/standard_512x512_40k_levircd.py']

checkpoint = None # noqa

model = dict(pretrained=checkpoint, decode_head=dict(channels=32,num_classes=2))

# optimizer
optimizer=dict(
    type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))