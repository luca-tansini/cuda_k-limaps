import plotly as py
import plotly.graph_objs as go
import sys

def readfile(filename):
    f = open(filename)
    out = []
    i = 0
    for l in f:
        if(i % 9 == 0): tmp = []
        if(i % 9 == 8):
            tmp += [float(l)]
            out += [tmp]
        else:
            tmp += [float(l)]
        i+=1
    return out

if __name__ == '__main__':

    if(len(sys.argv) != 5):
        print("usage: makeplotnoisy <cudaTimes> <pythonTimes> <outfile.html> <n>")
        exit(1)

    cuda = readfile(sys.argv[1])

    python = readfile(sys.argv[2])

    data = [
        go.Surface(z=cuda,name='CUDA',showscale=True,cmin=min(min(cuda)),cmax=max(max(python))),
        go.Surface(z=python,name='Python',opacity=0.8,showscale=False,cmin=min(min(cuda)),cmax=max(max(python)))
    ]

    layout = go.Layout(
        title = "k-LiMapS noisy test n = "+sys.argv[4],
        scene = {
            'zaxis':{
                'title':'avgTime (s)',
                'range':[min(min(cuda)),max(max(python))],
                'tickmode':'auto',
                'nticks':10
            },
            'yaxis':{
                'title':'δ (n/m)',
                'tickmode':'array',
                'tickvals':[0,1,2,3,4],
                'ticktext':['1.00','0.50','0.33','0.25','0.20']
            },
            'xaxis':{
                'title':'ρ (k/n)',
                'tickmode':'array',
                'tickvals':[0,1,2,3,4,5,6,7,8],
                'ticktext':['0.10','0.15','0.20','0.25','0.30','0.35','0.40','0.45','0.50']
            },
            'annotations':[{
                'x':8,
                'y':4,
                'z':cuda[4][8],
                'ax':0,
                'ay':0,
                'text':"CUDA",
                'arrowhead':0,
                'xanchor':"left",
                'yanchor':"bottom"
            },
            {
                'x':8,
                'y':4,
                'z':python[4][8],
                'ax':0,
                'ay':0,
                'text':"CPU",
                'arrowhead':0,
                'xanchor':"left",
                'yanchor':"bottom"
            }]
        }
    )

    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig,filename=sys.argv[3])
