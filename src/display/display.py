import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def get_cmap(type,encoded=True):
    if type.lower()=='vis':
        cmap,norm = vis_cmap(encoded)
        vmin,vmax=(0,10000) if encoded else (0,1)
    elif type.lower()=='vil':
        cmap,norm=vil_cmap(encoded)
        vmin,vmax=None,None
    elif type.lower()=='ir069':
        cmap,norm=c09_cmap(encoded)
        vmin,vmax=(-8000,-1000) if encoded else (-80,-10)
    elif type.lower()=='lght':
        cmap,norm='hot',None
        vmin,vmax=0,5
    else:
        cmap,norm='jet',None
        vmin,vmax=(-7000,2000) if encoded else (-70,20)

    return cmap,norm,vmin,vmax


def vil_cmap(encoded=True):
    cols=[   [0,0,0],
             [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
             [0.1568627450980392,  0.7450980392156863,  0.1568627450980392],
             [0.09803921568627451, 0.5882352941176471,  0.09803921568627451],
             [0.0392156862745098,  0.4117647058823529,  0.0392156862745098],
             [0.0392156862745098,  0.29411764705882354, 0.0392156862745098],
             [0.9607843137254902,  0.9607843137254902,  0.0],
             [0.9294117647058824,  0.6745098039215687,  0.0],
             [0.9411764705882353,  0.43137254901960786, 0.0],
             [0.6274509803921569,  0.0, 0.0],
             [0.9058823529411765,  0.0, 1.0]]
    lev = [16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]
    #TODO:  encoded=False
    nil = cols.pop(0)
    #nil=cols[0]
    under = cols[0]
    over = cols.pop()
    #over=cols[-1]
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap,norm
       
    
def vis_cmap(encoded=True):
    cols=[[0,0,0],
             [0.0392156862745098, 0.0392156862745098, 0.0392156862745098],
             [0.0784313725490196, 0.0784313725490196, 0.0784313725490196],
             [0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
             [0.1568627450980392, 0.1568627450980392, 0.1568627450980392],
             [0.19607843137254902, 0.19607843137254902, 0.19607843137254902],
             [0.23529411764705882, 0.23529411764705882, 0.23529411764705882],
             [0.27450980392156865, 0.27450980392156865, 0.27450980392156865],
             [0.3137254901960784, 0.3137254901960784, 0.3137254901960784],
             [0.35294117647058826, 0.35294117647058826, 0.35294117647058826],
             [0.39215686274509803, 0.39215686274509803, 0.39215686274509803],
             [0.43137254901960786, 0.43137254901960786, 0.43137254901960786],
             [0.47058823529411764, 0.47058823529411764, 0.47058823529411764],
             [0.5098039215686274, 0.5098039215686274, 0.5098039215686274],
             [0.5490196078431373, 0.5490196078431373, 0.5490196078431373],
             [0.5882352941176471, 0.5882352941176471, 0.5882352941176471],
             [0.6274509803921569, 0.6274509803921569, 0.6274509803921569],
             [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
             [0.7058823529411765, 0.7058823529411765, 0.7058823529411765],
             [0.7450980392156863, 0.7450980392156863, 0.7450980392156863],
             [0.7843137254901961, 0.7843137254901961, 0.7843137254901961],
             [0.8235294117647058, 0.8235294117647058, 0.8235294117647058],
             [0.8627450980392157, 0.8627450980392157, 0.8627450980392157],
             [0.9019607843137255, 0.9019607843137255, 0.9019607843137255],
             [0.9411764705882353, 0.9411764705882353, 0.9411764705882353],
             [0.9803921568627451, 0.9803921568627451, 0.9803921568627451],
             [0.9803921568627451, 0.9803921568627451, 0.9803921568627451]]
    lev=np.array([0.  , 0.02, 0.04, 0.06, 0.08, 0.1 , 0.12, 0.14, 0.16, 0.2 , 0.24,
       0.28, 0.32, 0.36, 0.4 , 0.44, 0.48, 0.52, 0.56, 0.6 , 0.64, 0.68,
       0.72, 0.76, 0.8 , 0.9 , 1.  ])
    if encoded:
        lev*=1e4
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap,norm


def ir_cmap(encoded=True):
    cols=[[0,0,0],[1.0, 1.0, 1.0],
     [0.9803921568627451, 0.9803921568627451, 0.9803921568627451],
     [0.9411764705882353, 0.9411764705882353, 0.9411764705882353],
     [0.9019607843137255, 0.9019607843137255, 0.9019607843137255],
     [0.8627450980392157, 0.8627450980392157, 0.8627450980392157],
     [0.8235294117647058, 0.8235294117647058, 0.8235294117647058],
     [0.7843137254901961, 0.7843137254901961, 0.7843137254901961],
     [0.7450980392156863, 0.7450980392156863, 0.7450980392156863],
     [0.7058823529411765, 0.7058823529411765, 0.7058823529411765],
     [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
     [0.6274509803921569, 0.6274509803921569, 0.6274509803921569],
     [0.5882352941176471, 0.5882352941176471, 0.5882352941176471],
     [0.5490196078431373, 0.5490196078431373, 0.5490196078431373],
     [0.5098039215686274, 0.5098039215686274, 0.5098039215686274],
     [0.47058823529411764, 0.47058823529411764, 0.47058823529411764],
     [0.43137254901960786, 0.43137254901960786, 0.43137254901960786],
     [0.39215686274509803, 0.39215686274509803, 0.39215686274509803],
     [0.35294117647058826, 0.35294117647058826, 0.35294117647058826],
     [0.3137254901960784, 0.3137254901960784, 0.3137254901960784],
     [0.27450980392156865, 0.27450980392156865, 0.27450980392156865],
     [0.23529411764705882, 0.23529411764705882, 0.23529411764705882],
     [0.19607843137254902, 0.19607843137254902, 0.19607843137254902],
     [0.1568627450980392, 0.1568627450980392, 0.1568627450980392],
     [0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
     [0.0784313725490196, 0.0784313725490196, 0.0784313725490196],
     [0.0392156862745098, 0.0392156862745098, 0.0392156862745098],
     [0.0, 0.803921568627451, 0.803921568627451]]
    lev=np.array([-110. , -105.2,  -95.2,  -85.2,  -75.2,  -65.2,  -55.2,  -45.2,
        -35.2,  -28.2,  -23.2,  -18.2,  -13.2,   -8.2,   -3.2,    1.8,
          6.8,   11.8,   16.8,   21.8,   26.8,   31.8,   36.8,   41.8,
         46.8,   51.8,   90. ,  100. ])
    if encoded:
        lev*=1e2
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap,norm         


def c09_cmap(encoded=True):
    cols=[
    [1.000000, 0.000000, 0.000000],
    [1.000000, 0.031373, 0.000000],
    [1.000000, 0.062745, 0.000000],
    [1.000000, 0.094118, 0.000000],
    [1.000000, 0.125490, 0.000000],
    [1.000000, 0.156863, 0.000000],
    [1.000000, 0.188235, 0.000000],
    [1.000000, 0.219608, 0.000000],
    [1.000000, 0.250980, 0.000000],
    [1.000000, 0.282353, 0.000000],
    [1.000000, 0.313725, 0.000000],
    [1.000000, 0.349020, 0.003922],
    [1.000000, 0.380392, 0.003922],
    [1.000000, 0.411765, 0.003922],
    [1.000000, 0.443137, 0.003922],
    [1.000000, 0.474510, 0.003922],
    [1.000000, 0.505882, 0.003922],
    [1.000000, 0.537255, 0.003922],
    [1.000000, 0.568627, 0.003922],
    [1.000000, 0.600000, 0.003922],
    [1.000000, 0.631373, 0.003922],
    [1.000000, 0.666667, 0.007843],
    [1.000000, 0.698039, 0.007843],
    [1.000000, 0.729412, 0.007843],
    [1.000000, 0.760784, 0.007843],
    [1.000000, 0.792157, 0.007843],
    [1.000000, 0.823529, 0.007843],
    [1.000000, 0.854902, 0.007843],
    [1.000000, 0.886275, 0.007843],
    [1.000000, 0.917647, 0.007843],
    [1.000000, 0.949020, 0.007843],
    [1.000000, 0.984314, 0.011765],
    [0.968627, 0.952941, 0.031373],
    [0.937255, 0.921569, 0.050980],
    [0.901961, 0.886275, 0.074510],
    [0.870588, 0.854902, 0.094118],
    [0.835294, 0.823529, 0.117647],
    [0.803922, 0.788235, 0.137255],
    [0.772549, 0.756863, 0.160784],
    [0.737255, 0.725490, 0.180392],
    [0.705882, 0.690196, 0.200000],
    [0.670588, 0.658824, 0.223529],
    [0.639216, 0.623529, 0.243137],
    [0.607843, 0.592157, 0.266667],
    [0.572549, 0.560784, 0.286275],
    [0.541176, 0.525490, 0.309804],
    [0.509804, 0.494118, 0.329412],
    [0.474510, 0.462745, 0.349020],
    [0.752941, 0.749020, 0.909804],
    [0.800000, 0.800000, 0.929412],
    [0.850980, 0.847059, 0.945098],
    [0.898039, 0.898039, 0.964706],
    [0.949020, 0.949020, 0.980392],
    [1.000000, 1.000000, 1.000000],
    [0.964706, 0.980392, 0.964706],
    [0.929412, 0.960784, 0.929412],
    [0.890196, 0.937255, 0.890196],
    [0.854902, 0.917647, 0.854902],
    [0.815686, 0.894118, 0.815686],
    [0.780392, 0.874510, 0.780392],
    [0.745098, 0.850980, 0.745098],
    [0.705882, 0.831373, 0.705882],
    [0.670588, 0.807843, 0.670588],
    [0.631373, 0.788235, 0.631373],
    [0.596078, 0.764706, 0.596078],
    [0.560784, 0.745098, 0.560784],
    [0.521569, 0.721569, 0.521569],
    [0.486275, 0.701961, 0.486275],
    [0.447059, 0.678431, 0.447059],
    [0.411765, 0.658824, 0.411765],
    [0.376471, 0.635294, 0.376471],
    [0.337255, 0.615686, 0.337255],
    [0.301961, 0.592157, 0.301961],
    [0.262745, 0.572549, 0.262745],
    [0.227451, 0.549020, 0.227451],
    [0.192157, 0.529412, 0.192157],
    [0.152941, 0.505882, 0.152941],
    [0.117647, 0.486275, 0.117647],
    [0.078431, 0.462745, 0.078431],
    [0.043137, 0.443137, 0.043137],
    [0.003922, 0.419608, 0.003922],
    [0.003922, 0.431373, 0.027451],
    [0.003922, 0.447059, 0.054902],
    [0.003922, 0.462745, 0.082353],
    [0.003922, 0.478431, 0.109804],
    [0.003922, 0.494118, 0.137255],
    [0.003922, 0.509804, 0.164706],
    [0.003922, 0.525490, 0.192157],
    [0.003922, 0.541176, 0.215686],
    [0.003922, 0.556863, 0.243137],
    [0.007843, 0.568627, 0.270588],
    [0.007843, 0.584314, 0.298039],
    [0.007843, 0.600000, 0.325490],
    [0.007843, 0.615686, 0.352941],
    [0.007843, 0.631373, 0.380392],
    [0.007843, 0.647059, 0.403922],
    [0.007843, 0.662745, 0.431373],
    [0.007843, 0.678431, 0.458824],
    [0.007843, 0.694118, 0.486275],
    [0.011765, 0.705882, 0.513725],
    [0.011765, 0.721569, 0.541176],
    [0.011765, 0.737255, 0.568627],
    [0.011765, 0.752941, 0.596078],
    [0.011765, 0.768627, 0.619608],
    [0.011765, 0.784314, 0.647059],
    [0.011765, 0.800000, 0.674510],
    [0.011765, 0.815686, 0.701961],
    [0.011765, 0.831373, 0.729412],
    [0.015686, 0.843137, 0.756863],
    [0.015686, 0.858824, 0.784314],
    [0.015686, 0.874510, 0.807843],
    [0.015686, 0.890196, 0.835294],
    [0.015686, 0.905882, 0.862745],
    [0.015686, 0.921569, 0.890196],
    [0.015686, 0.937255, 0.917647],
    [0.015686, 0.952941, 0.945098],
    [0.015686, 0.968627, 0.972549],
    [1.000000, 1.000000, 1.000000]]
    
    return ListedColormap(cols),None
