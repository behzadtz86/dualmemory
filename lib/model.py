__author__ = "Behzad Taghipour Zarfsaz"
__email__ = "behzad.taghipour-zarfsaz@informatik.hs-fulda.de"

import logging
import shutil
from math import exp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tqdm import trange
from lib.bqueue import Bqueue
from lib.dnn import Dnn
from lib.helper import Helper
from lib.som import SOM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model")
terminal_columns = shutil.get_terminal_size().columns // 2


class Model:
    def __init__(
            self, input_dim, batch_size, som_x, som_y,
            label_class_num, xt, tt, limit, stm
    ):
        self.input_dim = input_dim
        self.som_x = som_x
        self.som_y = som_y
        self.class_num = label_class_num
        self.batch_size = batch_size
        self.som = SOM(self.som_x, self.som_y, self.input_dim)
        self.dnn = Dnn(self.som_x * self.som_y, self.class_num)
        self.x_test = xt
        self.t_test = tt
        self.stm = Bqueue(max_size=stm)
        self.limit = limit
        self.scaler = StandardScaler()

    def transfer(self, dist):
        return self.scaler.fit_transform(dist)

    @staticmethod
    def flatten(samples):
        return np.reshape(samples, newshape=[-1, samples.shape[1] * samples.shape[2]])

    def reply(self):
        samples = None
        labels = None
        stm_samples = np.array([s[0] for s in self.stm.get_list()]).astype("float32")
        stm_labels = np.array([s[1] for s in self.stm.get_list()]).astype("float32")
        if stm_samples.shape[0] > 0:
            for i in trange(self.class_num, desc="Replying Data"):
                class_stm_idx = np.argwhere(np.argmax(stm_labels, axis=1) == i).ravel()
                if class_stm_idx.shape[0] == 0:
                    break
                class_prototypes = stm_samples[class_stm_idx]
                ll = stm_labels[class_stm_idx]
                g_samples = np.repeat(
                    class_prototypes, self.limit // class_prototypes.shape[0], axis=0
                )
                g_labels = np.repeat(ll, self.limit // class_prototypes.shape[0], axis=0)
                if i == 0:
                    samples = g_samples
                    labels = g_labels
                else:
                    samples = np.concatenate((samples, g_samples))
                    labels = np.concatenate((labels, g_labels))
        return samples, labels

    def fill_stm(self, samples, z_som, labels):
        logger.info("\rFilling STM")
        loss, _ = self.dnn.evaluate(z_som, labels, batch_size=1, verbose=0)
        loss = np.array(loss).astype("float32")
        stm_idx = np.argwhere(loss > 0.1).ravel()
        if stm_idx.shape[0] == 0:
            wrong_samples, wrong_labels = Helper.get_random_samples(samples, labels, self.limit)
        else:
            wrong_samples = samples[stm_idx]
            wrong_labels = labels[stm_idx]
        for s in range(self.class_num):
            class_idx = np.argwhere(np.argmax(wrong_labels, axis=1) == s).ravel()
            loop_iter = min(self.stm.max_size // self.class_num, class_idx.shape[0])
            for i in range(loop_iter):
                self.stm.push(
                    (wrong_samples[class_idx[i]], wrong_labels[class_idx[i]])
                )

    def train(
            self, samples, labels, dnn_iter, som_lr, som_rad, ce, sub_task
    ):
        samples, labels = shuffle(samples, labels)
        logger.info("\r".center(terminal_columns, "="))
        logger.info(f"\r Sub-Task D{sub_task}")
        logger.info("\r".center(terminal_columns, "="))
        if sub_task > 1:
            m_samples, m_labels = self.reply()
            if m_samples is not None:
                samples = np.concatenate((samples, m_samples))
                labels = np.concatenate((labels, m_labels))
                samples, labels = shuffle(samples, labels)
        x, t = Helper.generate_batches(samples, labels, self.batch_size)
        sigma = []
        confusion_matrices = []
        # cm_list = range(len(x))
        cm = []
        pbar = trange(len(x))
        for i in pbar:
            decay = exp(-1 * (5 * i / len(x)))
            sigma.append(som_rad * decay)
            z_som = self.transfer(self.som.get_distances(x[i]))
            loss, acc = self.dnn.evaluate(z_som, t[i], verbose=0)
            loss = np.array(loss)
            wrong_idx = np.argwhere(np.greater(np.array(loss), ce)).ravel()
            if wrong_idx.shape[0] > 0:
                wrong_samples = x[i][wrong_idx]
                self.som.train(
                    wrong_samples, learning_rate=som_lr * decay,
                    radius=som_rad * decay, global_order=self.batch_size
                )
            z_som = self.transfer(self.som.get_distances(x[i], batch_size=self.batch_size))
            z_som_test = self.transfer(
                self.som.get_distances(self.x_test, batch_size=self.batch_size)
            )
            cm = i in cm_list and sub_task > 1
            d_loss, d_acc, confusion_matrix = self.dnn.train(
                z_som, t[i], z_som_test, self.t_test,
                cm=cm, epoch=dnn_iter, batch_size=self.batch_size
            )
            if len(confusion_matrix) > 0:
                for m in confusion_matrix:
                    confusion_matrices.append(m)
            d_acc = np.mean(np.array(d_acc).astype("float32"))
            pbar.set_description(
                f"Batch:{i + 1}/{len(x)}|Train Acc.:{d_acc:.4f}"
            )
            pbar.refresh()
        logger.info("\rFinal DNN Training & Evaluation...")
        z_som_test = self.transfer(self.som.get_distances(self.x_test, batch_size=self.batch_size))
        z_som_stm = self.transfer(self.som.get_distances(samples, batch_size=self.batch_size))
        _, _, confusion_matrix = self.dnn.train(
            z_som_stm, labels, z_som_test, self.t_test,
            cm=True, epoch=dnn_iter, batch_size=self.batch_size
        )
        confusion_matrices.append(confusion_matrix[0])
        loss, accuracy = self.dnn.evaluate(z_som_test, self.t_test, verbose=1)
        self.fill_stm(samples, z_som_stm, labels)
        return accuracy, np.array(sigma), confusion_matrices
