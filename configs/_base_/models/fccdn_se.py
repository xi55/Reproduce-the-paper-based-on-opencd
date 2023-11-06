norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

model=dict(
    type='DIEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='FCCDN',
        num_band=3,
        os=16,
        use_se=False, 
        dilation_list = [1, 1, 1, 1],
        stride_list = [2, 2, 2, 2],
        pool_list = [True, True, True, True],
        channel_list = [256, 128, 64, 32],
    ),
    neck=dict(
        type='NL_FPN',
        channel_list = [256, 128, 64, 32]
    ),
    decode_head=dict(
        type='Fccdn_Head',
        in_channels = 3,
        channels=32,
        num_classes=2,
        channel_list = [256, 128, 64, 32],
        pool_list = [True, True, True, True],
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='FCCDN_loss_BCD', loss_name='fccdn_loss', loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)