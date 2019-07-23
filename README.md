# mnist.rs

[![Build Status](https://travis-ci.com/skyzh/mnist.rs.svg?branch=master)](https://travis-ci.com/skyzh/mnist.rs)

Train neural network for MNIST database implemented in Rust

## Usage

Firstly, put MNIST dataset in `/data` folder:    
```
data/
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
```

Then,     
```bash
cargo run --release
```

With current setting of `784 x 30 x 10 network`, `mini batch = 10`, `learning speed = 1.0`, the result is:    
```
Epoch 0: 9101/10000
Epoch 1: 9229/10000
Epoch 2: 9313/10000
Epoch 3: 9357/10000
Epoch 4: 9396/10000
Epoch 5: 9411/10000
Epoch 6: 9434/10000
Epoch 7: 9440/10000
Epoch 8: 9448/10000
Epoch 9: 9452/10000
Epoch 10: 9476/10000
Epoch 11: 9499/10000
Epoch 12: 9510/10000
Epoch 13: 9506/10000
Epoch 14: 9474/10000
Epoch 15: 9507/10000
Epoch 16: 9491/10000
Epoch 17: 9502/10000
Epoch 18: 9507/10000
Epoch 19: 9509/10000
Epoch 20: 9503/10000
Epoch 21: 9506/10000
Epoch 22: 9508/10000
Epoch 23: 9495/10000
Epoch 24: 9493/10000
Epoch 25: 9531/10000
Epoch 26: 9513/10000
Epoch 27: 9522/10000
Epoch 28: 9506/10000
Epoch 29: 9500/10000
Epoch 30: 9530/10000
Epoch 31: 9521/10000
Epoch 32: 9531/10000
Epoch 33: 9527/10000
Epoch 34: 9537/10000
Epoch 35: 9521/10000
Epoch 36: 9529/10000
Epoch 37: 9543/10000
Epoch 38: 9534/10000
Epoch 39: 9526/10000
Epoch 40: 9543/10000
```