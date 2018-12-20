import numpy as np
from keras import initializers
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, concatenate, multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import DataSet
from time import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[256,256,128,64]',
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    return parser.parse_args()


def get_model(num_users, num_items, layers):
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name='user_embedding',
                                   input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='item_embedding',
                                   input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))

    # The 0-th layer is the concatenation of embedding layers
    # vector = merge([user_latent, item_latent], mode = 'concat')
    vector = concatenate([user_latent, item_latent])
    # attention_probs = Dense(64, activation='softmax', name='attention_vec')(vector)
    # vector = multiply([vector, attention_probs], name='attention_mul')
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], activation='relu', name='layer%d' % idx)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer=initializers.lecun_normal(),
                       name='prediction')(vector)

    model_ = Model(inputs=[user_input, item_input],
                   outputs=prediction)
    print(model_.summary())
    return model_


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            # while train.has_key((u, j)):
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    num_negatives = args.num_neg
    batch_size = args.batch_size
    epochs = args.epochs

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("MLP arguments: %s " % args)
    model_out_file = '../Pretrain/%s_MLP_%s_%d.h5' % (args.dataset, args.layers, time())

    # Loading data
    t1 = time()
    dataset = DataSet(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, layers)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')


    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        # Evaluation
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
