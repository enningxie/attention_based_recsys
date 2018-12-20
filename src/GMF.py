import numpy as np
import argparse
from keras.models import Input, Model
from keras.layers import Embedding, Flatten, multiply, Dense
from keras import initializers, optimizers, losses
from Dataset import DataSet
from evaluate import evaluate_model
from time import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='../Data/', type=str, help='Input data path.')
    parser.add_argument('--data_set', default='ml-1m', type=str)
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs.')
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--num_factors', type=int, default=64, help='Embedding size.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    return parser.parse_args()


def GMF(num_users, num_items, latent_dim):
    # input
    user_input = Input(shape=(1,), dtype='int32', name='user_input_op')
    item_input = Input(shape=(1,), dtype='int32', name='item_input_op')

    # embedding
    user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim,
                               embeddings_initializer=initializers.random_normal(),
                               input_length=1, name='user_embedding_op')(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=latent_dim,
                               embeddings_initializer=initializers.random_normal(),
                               input_length=1, name='item_embedding_op')(item_input)

    # flatten
    user_flatten = Flatten()(user_embedding)
    item_flatten = Flatten()(item_embedding)

    # multiply
    predict_vector = multiply([user_flatten, item_flatten])

    # prediction
    prediction = Dense(1, activation='sigmoid', name='prediction_op')(predict_vector)

    model_ = Model(inputs=[user_input, item_input], outputs=prediction)

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
    num_factors = args.num_factors
    num_negatives = args.num_neg
    epochs = args.epochs
    batch_size = args.batch_size
    topK = 10
    threads = 1

    # loading data
    t1 = time()
    dataSet = DataSet(args.path + args.data_set)
    train, testRatings, testNegatives = dataSet.trainMatrix, dataSet.testRatings, dataSet.testNegatives
    num_users, num_items = dataSet.num_users, dataSet.num_items

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = GMF(num_users, num_items, num_factors)
    # compile
    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.binary_crossentropy)

    # Init performance
    t2 = time()
    # xz
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t2))

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
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, threads)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
        if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch


    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
