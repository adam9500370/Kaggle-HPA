from models.resnet import *


def get_model(name, n_classes, load_pretrained=True, use_cbam=False):
    model = _get_model_instance(name)

    if 'resnet' in name:
        model = model(name=name, n_classes=n_classes, load_pretrained=load_pretrained, use_cbam=use_cbam)

    return model

def _get_model_instance(name):
    try:
        return {
            'resnet18': resnet,
            'resnet34': resnet,
            'resnet50': resnet,
            'resnet101': resnet,
        }[name]
    except:
        print('Model {} not available'.format(name))
