# -*- coding: utf-8 -*-
from utils.profiling import load_profile_model

# FireNet
# profile_model('firenet')
# OctFiResNet
load_profile_model('octfiresnet')
# ResNet50
load_profile_model('resnet')
# KutralNet
load_profile_model('kutralnet')
# KutralNet Mobile
load_profile_model('kutralnet_mobile')
# KutralNetOctave no-groups
load_profile_model('kutralnetoct', extra_params={'groups':False})
# KutralNetOctave
load_profile_model('kutralnetoct')
# KutralNet MobileOctave
load_profile_model('kutralnet_mobileoct')