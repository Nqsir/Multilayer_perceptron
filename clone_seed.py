from lib.save import load_object, save_object
from lib.Network import Network
from lib.FCLayer import FCLayer
from lib.ActivationLayer import ActivationLayer
from lib.Activations import softmax, softmax_prime, tanh, tanh_prime


def clone():
    tmp = load_object("seed-9473.save")

    net = Network()

    net.add(FCLayer(30, 30))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(30, 30))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(30, 30))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(30, 2))
    net.add(ActivationLayer(softmax, softmax_prime))

    net.layers[0] = tmp.layers[0]
    net.layers[1] = tmp.layers[1]
    net.layers[2] = tmp.layers[2]
    net.layers[3] = tmp.layers[3]
    net.layers[4] = tmp.layers[4]
    net.layers[5] = tmp.layers[5]
    net.layers[6] = tmp.layers[6]

    save_object(net, f'seed-clone-9473-soft.save')


if __name__ == '__main__':
    clone()
