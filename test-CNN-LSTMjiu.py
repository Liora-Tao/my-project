import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import glob
# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, metrics

# Scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


import warnings
warnings.filterwarnings('ignore')
from scipy.interpolate import CubicSpline
import joblib
import json
import argparse
import os
tf.compat.v1.enable_eager_execution()


def focal_loss(gamma=2.0, alpha=0.25):

    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        

        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        

        alpha_t = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_t, 1 - alpha_t)
        
        cross_entropy = -tf.math.log(p_t)
        weight = alpha_t * tf.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        
        return tf.reduce_mean(loss)
    
    return focal_loss_fixed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, default="train", help="run mode:train,test,train_now")
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--data_path", type=str, default='D:\work\\project code\\Classification\\train_files', help="train data path")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default='D:\\work\\project code\\Classification\\curve_classifier', help="load resume")
    parser.add_argument("--model_type", type=str, default="cnn_lstm", help="model_type")
    parser.add_argument("--checkpoint", type=str, default="sam-med2d_b.pth", help="checkpoint")
    parser.add_argument("--model_name", type=str, default="curve_classifier", help="model name")
    parser.add_argument("--use_focal_loss", type=bool, default=True, help="use focal loss")
    parser.add_argument("--use_data_aug", type=bool, default=False, help="use data augmentation")

    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None

    args_dict = vars(args)
    return args

def read_one_file(path):

    df = pd.read_excel(path)
    X = []
    Y = []
    for key,value in df.iterrows():
        points = get_data(value['ori_point'])
        type = value['type']
        X.append(np.array(points).reshape(-1,1))
        Y.append(1 if type==1 else 0)
    return X, Y

def read_many_files(path):

    files = glob.glob(path + '/*.xlsx')
    X = []
    Y = []
    for file in files:
        df = pd.read_excel(file)
        for key,value in df.iterrows():
            points = get_data(value['ori_point'])
            if len(points) != 50:

                new_index = np.linspace(0, len(points) - 1, 50)

                new_values = np.interp(new_index, np.arange(len(points)), points.values)
                points = pd.Series(new_values, index=range(50))

            type = value['type']
            X.append(np.array(points).reshape(-1, 1))
            Y.append(1 if type==1 else 0)
    return X, Y

def get_data(data):

    data = data.split('\n')
    point_list = []
    for item in data:
        item = item.split('[')[-1]
        item = item.split(']')[0]
        item = item.split(',')
        for point in item:
            if len(point) > 0:
                point_list.append(float(point))
    return pd.Series(point_list)

class DeepCurveClassifier:

    def __init__(self,
                 input_shape: Tuple[int, int],
                 model_type: str = 'cnn_lstm_hybrid',
                 random_seed: int = 42,
                 args: dict = {}):

        self.input_shape = input_shape
        self.model_type = model_type
        self.random_seed = random_seed

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        self.threshold = 0.87


        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    def build_cnn_model(self) -> keras.Model:

        inputs = keras.Input(shape=self.input_shape)


        x1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.2)(x1)
        x1 = layers.MaxPooling1D(pool_size=2)(x1)


        x2 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x1)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.3)(x2)
        x2 = layers.MaxPooling1D(pool_size=2)(x2)


        x3 = layers.Conv1D(256, kernel_size=5, padding='same', activation='relu')(x2)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Dropout(0.3)(x3)
        

        x4 = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x3)
        x4 = layers.BatchNormalization()(x4)
        x4 = layers.Dropout(0.4)(x4)


        avg_pool = layers.GlobalAveragePooling1D()(x4)
        max_pool = layers.GlobalMaxPooling1D()(x4)
        x = layers.Concatenate()([avg_pool, max_pool])


        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)


        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_Classifier")
        return model

    def build_cnn_lstm_model(self) -> keras.Model:

        inputs = keras.Input(shape=self.input_shape)


        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)


        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
        x = layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.1))(x)


        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)


        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Classifier")
        return model

    def build_attention_model(self) -> keras.Model:

        inputs = keras.Input(shape=self.input_shape)


        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)


        for _ in range(2):

            attention_output = layers.MultiHeadAttention(
                num_heads=8,
                key_dim=32,
                dropout=0.1
            )(x, x)


            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)


            ffn = layers.Dense(128, activation='relu')(x)
            ffn = layers.Dropout(0.2)(ffn)
            ffn = layers.Dense(64)(ffn)
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization()(x)


        attention_weights = layers.Dense(1, activation='softmax')(x)
        x = layers.Multiply()([x, attention_weights])
        x = layers.GlobalAveragePooling1D()(x)


        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)


        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="Attention_Classifier")
        return model

    def build_resnet_model(self) -> keras.Model:

        inputs = keras.Input(shape=self.input_shape)


        x = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=3)(x)


        def residual_block(x_in, filters, kernel_size=3):

            x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x_in)
            x = layers.BatchNormalization()(x)
            x = layers.Conv1D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)


            if x_in.shape[-1] != filters:
                x_in = layers.Conv1D(filters, 1)(x_in)

            x = layers.Add()([x, x_in])
            x = layers.Activation('relu')(x)
            return x


        x = residual_block(x, 64)
        x = residual_block(x, 64)

        x = layers.Conv1D(128, kernel_size=3, strides=2, padding='same')(x)
        x = residual_block(x, 128)
        x = residual_block(x, 128)


        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet_Classifier")
        return model

    def build_model(self, args: dict = {}) -> keras.Model:

        if self.model_type == 'cnn':
            model = self.build_cnn_model()
        elif self.model_type == 'cnn_lstm':
            model = self.build_cnn_lstm_model()
        elif self.model_type == 'attention':
            model = self.build_attention_model()
        elif self.model_type == 'resnet':
            model = self.build_resnet_model()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")


        if isinstance(args, dict):
            lr = args.get('lr', 1e-4)
            use_focal_loss = args.get('use_focal_loss', True)
        else:
            lr = getattr(args, 'lr', 1e-4)
            use_focal_loss = getattr(args, 'use_focal_loss', True)
        

        optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        

        if use_focal_loss:
            loss_fn = focal_loss(gamma=2.0, alpha=0.25)
        else:
            loss_fn = 'binary_crossentropy'

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                'accuracy',
                metrics.AUC(name='auc'),
                metrics.Precision(name='precision'),
                metrics.Recall(name='recall'),
                metrics.AUC(name='pr_auc', curve='PR')  # PR曲线下面积
            ]
        )

        self.model = model
        return model

    def prepare_data(self,
                     X: np.ndarray,
                     y: Optional[np.ndarray] = None,
                     fit_scaler: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:


        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)


        if fit_scaler:

            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
        else:
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)

        if y is not None:

            y = np.array(y).flatten()
            return X_scaled, y
        else:
            return X_scaled

    def create_class_weights(self, y: np.ndarray) -> Dict[int, float]:

        unique, counts = np.unique(y, return_counts=True)
        total = len(y)

        class_weights = {}
        for cls, count in zip(unique, counts):

            class_weights[int(cls)] = total / (len(unique) * count)


        if 0 in class_weights and 1 in class_weights:

            class_weights[0] = class_weights[0] * 1.2

        return class_weights

    def train(self, X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              args: dict = {},
              validation_split: float = 0.2,
              use_class_weights: bool = True) -> keras.callbacks.History:


        X_train_processed, y_train_processed = self.prepare_data(X_train, y_train, fit_scaler=True)


        if self.model is None:
            self.build_model(args=args)


        monitor_metric = 'val_auc' if X_val is not None else 'auc'
        monitor_loss = 'val_loss' if X_val is not None else 'loss'
        

        if isinstance(args, dict):
            work_dir = args.get('work_dir', 'workdir')
            lr = args.get('lr', 1e-4)
            epochs = args.get('epochs', 50)
        else:
            work_dir = getattr(args, 'work_dir', 'workdir')
            lr = getattr(args, 'lr', 1e-4)
            epochs = getattr(args, 'epochs', 50)
        

        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'checkpoints'), exist_ok=True)
        

        def lr_schedule(epoch):

            return lr * (0.5 * (1 + np.cos(np.pi * epoch / max(epochs, 1))))
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor=monitor_loss,
                patience=10,
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-4
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor_loss,
                factor=0.5,
                patience=5,
                min_lr=lr * 0.01,
                verbose=1,
                mode='min'
            ),
            callbacks.ModelCheckpoint(
                os.path.join(work_dir, 'checkpoints', 'best_model.h5'),
                monitor=monitor_metric,
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            ),

            callbacks.LearningRateScheduler(lr_schedule, verbose=0)
        ]


        class_weights = None
        if use_class_weights:
            class_weights = self.create_class_weights(y_train_processed)
            print(f"Category weight: {class_weights}")


        if X_val is not None and y_val is not None:
            X_val_processed, y_val_processed = self.prepare_data(X_val, y_val, fit_scaler=False)
            validation_data = (X_val_processed, y_val_processed)
        else:
            validation_data = None

        self.history = self.model.fit(
            X_train_processed,
            y_train_processed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else None,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=1
        )

        return self.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        X_processed = self.prepare_data(X, fit_scaler=False)
        probabilities = self.model.predict(X_processed, verbose=0)
        return probabilities.flatten()

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:

        if threshold is None:
            threshold = self.threshold

        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)

        return predictions, probabilities

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:


        y_pred, y_proba = self.predict(X_test)


        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)


        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)


        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        avg_precision = average_precision_score(y_test, y_proba)


        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = _[optimal_idx] if len(_) > optimal_idx else 0.5

        results = {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'average_precision': avg_precision,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'optimal_threshold': optimal_threshold,
            'predictions': y_pred,
            'probabilities': y_proba,
            'accuracy': report['accuracy'],
            'precision_positive': report.get('1', {}).get('precision', 0),
            'recall_positive': report.get('1', {}).get('recall', 0),
            'f1_positive': report.get('1', {}).get('f1-score', 0)
        }

        return results

    def analyze_rain_points(self,
                            X: np.ndarray,
                            probabilities: np.ndarray,
                            uncertainty_threshold: float = 0.2) -> Dict:

        uncertainty = np.abs(probabilities - 0.5)
        is_rain_point = uncertainty < uncertainty_threshold


        confidence = 1 - (2 * uncertainty)


        categories = []
        for prob, conf in zip(probabilities, confidence):
            if conf < 0.4:
                if prob > 0.5:
                    categories.append("Low confidence positive")
                else:
                    categories.append("Low confidence negative")
            elif prob > 0.7:
                categories.append("High confidence positive")
            elif prob < 0.3:
                categories.append("High confidence negative")
            else:
                categories.append("Boundary sample")


        recommendations = []
        for prob, conf, category in zip(probabilities, confidence, categories):
            if "Boundary" in category or "Low confidence" in category:
                if prob > 0.5:
                    rec = f"Suspected positive result（Probability: {prob:.3f}），it is recommended to review or conduct a new experiment."
                else:
                    rec = f"Suspected negative result（Probability: {prob:.3f}），it is recommended to review or conduct a new experiment."
            elif "High confidence positive" in category:
                rec = f"Clarify the positive aspect（Probability: {prob:.3f}）"
            else:
                rec = f"Clarify the negative aspect（Probability: {prob:.3f}）"
            recommendations.append(rec)

        return {
            'is_rain_point': is_rain_point,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'categories': categories,
            'recommendations': recommendations,
            'rain_indices': np.where(is_rain_point)[0].tolist(),
            'rain_count': np.sum(is_rain_point),
            'rain_percentage': np.mean(is_rain_point) * 100
        }

    def calibrate_threshold(self,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            target_metric: str = 'f1') -> float:

        y_proba = self.predict_proba(X_val)

        thresholds = np.linspace(0.1, 0.9, 100)
        best_threshold = 0.5
        best_score = 0

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)


            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()


            if target_metric == 'f1':
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                if precision + recall > 0:
                    score = 2 * precision * recall / (precision + recall)
                else:
                    score = 0
            elif target_metric == 'precision':
                score = tp / (tp + fp) if (tp + fp) > 0 else 0
            elif target_metric == 'recall':
                score = tp / (tp + fn) if (tp + fn) > 0 else 0
            elif target_metric == 'specificity':
                score = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                raise ValueError(f"Unsupportable target indicators: {target_metric}")

            if score > best_score:
                best_score = score
                best_threshold = thresh

        self.threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.3f}, {target_metric}Score: {best_score:.3f}")

        return best_threshold

    def save_model(self, filepath: str):


        self.model.save(f"{filepath}_model.h5")


        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")

        config = {
            'input_shape': self.input_shape,
            'model_type': self.model_type,
            'threshold': self.threshold,
            'random_seed': self.random_seed
        }

        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(config, f)

        print(f"The model has been saved to: {filepath}_*")

    def load_model(self, filepath: str):


        self.model = keras.models.load_model(f"{filepath}_model.h5")


        self.scaler = joblib.load(f"{filepath}_scaler.pkl")

        with open(f"{filepath}_config.json", 'r') as f:
            config = json.load(f)

        self.input_shape = tuple(config['input_shape'])
        self.model_type = config['model_type']
        self.threshold = config['threshold']
        self.random_seed = config['random_seed']

        print(f"The model has been loaded from {filepath}_* ")

    def predict_from_file(self, file_path: str) -> Dict:


        files = glob.glob(file_path + '/*.xlsx')
        results = []

        for file in files:
            df = pd.read_excel(file)
            X = []
            Y = []


            original_indices = []

            for key, value in df.iterrows():
                points = get_data(value['ori_point'])
                if len(points) != 50:

                    new_index = np.linspace(0, len(points) - 1, 50)

                    new_values = np.interp(new_index, np.arange(len(points)), points.values)
                    points = pd.Series(new_values, index=range(50))

                type = value['type']
                X.append(np.array(points).reshape(-1, 1))
                Y.append(1 if type == 1 else 0)
                original_indices.append(key)

            X = np.array(X)


            processor = CurveDataProcessor()
            X_norm = processor.normalize_curves(X, method='minmax')


            predictions, probabilities = self.predict(X_norm)


            df['predict'] = predictions
            df['predict_probability'] = probabilities


            df.to_excel(file, index=False)

            result = {
                'predictions': predictions,
                'probabilities': probabilities,
                'X_original': X,
                'X_normalized': X_norm,
                'original_df': df,
                'file_path': file
            }

            results.append(result)

            print(f"预测结果已保存到: {file}")

        return results

    def train_from_directory(self, directory_path: str, args: dict):

        X, y = read_many_files(directory_path)
        X = np.array(X)


        processor = CurveDataProcessor()
        X_norm = processor.normalize_curves(X, method='minmax')


        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )


        history = self.train(
            X_train, y_train,
            X_val=X_test,
            y_val=y_test,
            args=args,
            use_class_weights=True
        )

        return history


class CurveDataProcessor:


    @staticmethod
    def augment_time_series(X: np.ndarray,
                            y: np.ndarray,
                            augment_factor: int = 3,
                            balance_classes: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        if not balance_classes:

            X_augmented = [X]
            y_augmented = [y]

            for i in range(augment_factor):

                noise = np.random.normal(0, 0.05, X.shape)
                X_noisy = X + noise
                X_augmented.append(X_noisy)
                y_augmented.append(y)


                X_warped = CurveDataProcessor.time_warp(X)
                X_augmented.append(X_warped)
                y_augmented.append(y)

            X_final = np.vstack(X_augmented)
            y_final = np.hstack(y_augmented)

            return X_final, y_final


        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))

        print(f"Original category distribution: {class_counts}")


        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)

        minority_count = class_counts[minority_class]
        majority_count = class_counts[majority_class]


        target_count = majority_count*0.6
        samples_needed = target_count - minority_count

        if samples_needed <= 0:

            return X, y


        minority_mask = (y == minority_class)
        majority_mask = (y == majority_class)

        X_minority = X[minority_mask]
        y_minority = y[minority_mask]
        X_majority = X[majority_mask]
        y_majority = y[majority_mask]


        samples_per_minority = int(max(1, samples_needed // len(y_minority)))
        X_augmented_minority = []
        y_augmented_minority = []


        for i in range(samples_per_minority):

            indices = np.random.choice(len(y_minority), len(y_minority), replace=True)
            X_selected = X_minority[indices]
            y_selected = y_minority[indices]


            augmented_data = CurveDataProcessor._apply_augmentation_transforms(X_selected)
            X_augmented_minority.append(augmented_data)
            y_augmented_minority.append(y_selected)


        if X_augmented_minority:
            X_minority_balanced = np.vstack([X_minority] + X_augmented_minority)
            y_minority_balanced = np.hstack([y_minority] + y_augmented_minority)
        else:
            X_minority_balanced = X_minority
            y_minority_balanced = y_minority


        X_final = np.vstack([X_minority_balanced, X_majority])
        y_final = np.hstack([y_minority_balanced, y_majority])

        final_counts = dict(zip(*np.unique(y_final, return_counts=True)))
        print(f"Enhanced category distribution: {final_counts}")

        return X_final, y_final

    @staticmethod
    def _apply_augmentation_transforms(X: np.ndarray) -> np.ndarray:

        transform_choice = np.random.choice(['noise', 'scale'])

        if transform_choice == 'noise':

            noise = np.random.normal(0, 0.05, X.shape)
            return X + noise
        elif transform_choice == 'scale':

            scale_factors = np.random.uniform(0.8, 1.2, (X.shape[0], 1, 1))
            return X * scale_factors


    @staticmethod
    def time_warp(X: np.ndarray, sigma: float = 0.1) -> np.ndarray:

        X_warped = np.zeros_like(X)
        n_timesteps = X.shape[1]

        for i in range(X.shape[0]):

            orig_time = np.arange(n_timesteps)


            random_warp = np.random.normal(1, sigma, n_timesteps)
            cum_warp = np.cumsum(random_warp)
            warp_time = np.interp(
                orig_time,
                np.linspace(0, n_timesteps - 1, n_timesteps),
                cum_warp
            )


            warp_time = np.clip(warp_time, 0, n_timesteps - 1)


            for j in range(X.shape[2]):
                X_warped[i, :, j] = np.interp(orig_time, warp_time, X[i, :, j])

        return X_warped

    @staticmethod
    def detect_outliers(X: np.ndarray, threshold: float = 3.0) -> np.ndarray:

        X_flat = X.reshape(X.shape[0], -1)
        mean = np.mean(X_flat, axis=0)
        std = np.std(X_flat, axis=0)
        std = np.where(std == 0, 1e-10, std)

        z_scores = np.abs((X_flat - mean) / std)
        outlier_mask = np.any(z_scores > threshold, axis=1)

        return outlier_mask

    @staticmethod
    def normalize_curves(X: np.ndarray, method: str = 'minmax') -> np.ndarray:

        X_norm = X.copy()

        if method == 'minmax':

            for i in range(X.shape[0]):
                for j in range(X.shape[2]):
                    min_val = np.min(X[i, :, j])
                    max_val = np.max(X[i, :, j])
                    if max_val > min_val:
                        X_norm[i, :, j] = (X[i, :, j] - min_val) / (max_val - min_val)
                    else:

                        X_norm[i, :, j] = 0.0

        elif method == 'zscore':

            for i in range(X.shape[0]):
                for j in range(X.shape[2]):
                    mean_val = np.mean(X[i, :, j])
                    std_val = np.std(X[i, :, j])
                    if std_val > 1e-8:
                        X_norm[i, :, j] = (X[i, :, j] - mean_val) / std_val
                    else:
                        X_norm[i, :, j] = 0.0
        elif method == 'robust':

            for i in range(X.shape[0]):
                for j in range(X.shape[2]):
                    median_val = np.median(X[i, :, j])
                    q75 = np.percentile(X[i, :, j], 75)
                    q25 = np.percentile(X[i, :, j], 25)
                    iqr = q75 - q25
                    if iqr > 1e-8:
                        X_norm[i, :, j] = (X[i, :, j] - median_val) / iqr
                    else:
                        X_norm[i, :, j] = 0.0

        return X_norm


class CurveVisualizer:


    def plot_learning_curves(self, history, figsize=(12, 8)):

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall', 'pr_auc']
        titles = ['loss function', 'precision', 'ROC AUC', 'accurate rate', 'recall rate', 'PR AUC']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 3, i % 3]


            if metric in history.history:
                ax.plot(history.history[metric], label='Train')


            val_metric = f'val_{metric}'
            if val_metric in history.history:
                ax.plot(history.history[val_metric], label='Test')

            ax.set_title(title)
            ax.set_xlabel('Train round')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./images/learning_curve.png')
        # plt.show()

    def plot_classification_results(self, results: Dict, figsize: Tuple[int, int] = (15, 10)):

        fig, axes = plt.subplots(2, 3, figsize=figsize)


        ax = axes[0, 0]
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('confusion matrix')
        ax.set_xlabel('predict lable')
        ax.set_ylabel('true lable')


        ax = axes[0, 1]
        fpr, tpr = results['fpr'], results['tpr']
        roc_auc = results['roc_auc']
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', lw=2)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('fake positive rate')
        ax.set_ylabel('true positive rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)


        ax = axes[0, 2]
        precision, recall = results['precision'][:-1], results['recall'][:-1]
        pr_auc = results['pr_auc']
        avg_precision = results['average_precision']
        ax.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('recall rate')
        ax.set_ylabel('accurate rate')
        ax.set_title('accurate-recall curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)


        ax = axes[1, 0]
        y_proba = results['probabilities']
        y_true = np.array(results.get('true_labels', None))

        if y_true is not None:
            ax.hist(y_proba[y_true == 0], bins=30, alpha=0.5, label='negative', color='blue')
            ax.hist(y_proba[y_true == 1], bins=30, alpha=0.5, label='positive', color='red')
        else:
            ax.hist(y_proba, bins=30, alpha=0.7, color='green')

        ax.axvline(0.5, color='black', linestyle='--', label='Decision boundary')
        ax.set_xlabel('prediction probability')
        ax.set_ylabel('sample size')
        ax.set_title('Prediction probability distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)


        ax = axes[1, 1]
        ax.axis('off')
        report_text = "report:\n\n"

        if 'classification_report' in results:
            report = results['classification_report']
            for key, value in report.items():
                if isinstance(value, dict):
                    report_text += f"{key}:\n"
                    for subkey, subvalue in value.items():
                        report_text += f"  {subkey}: {subvalue:.3f}\n"
                else:
                    report_text += f"{key}: {value:.3f}\n"

        ax.text(0.1, 0.9, report_text, fontsize=10, va='top',
                family='monospace', transform=ax.transAxes)


        ax = axes[1, 2]
        if 'threshold_analysis' in results:
            thresholds = results['threshold_analysis']['thresholds']
            f1_scores = results['threshold_analysis']['f1_scores']
            ax.plot(thresholds, f1_scores, 'b-', lw=2)
            optimal_idx = np.argmax(f1_scores)
            ax.axvline(thresholds[optimal_idx], color='r', linestyle='--',
                       label=f'best threshold: {thresholds[optimal_idx]:.3f}')
            ax.set_xlabel('threshold')
            ax.set_ylabel('F1 score')
            ax.set_title('Threshold optimization curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Threshold optimization curve',
                    ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig('./images/classification_results.png')

    def plot_sample_curves(self,
                           X: np.ndarray,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_proba: np.ndarray,
                           n_samples: int = 6,
                           figsize: Tuple[int, int] = (15, 10)):

        fig, axes = plt.subplots(1, 1, figsize=figsize)


        for idx in range(len(X)):
            curve = X[idx]


            true_label = y_true[idx] if idx < len(y_true) else 'Not known'
            pred_label = y_pred[idx] if idx < len(y_pred) else 'Not known'
            proba = y_proba[idx] if idx < len(y_proba) else 0

            if true_label == 1:
                color = 'green'
                linestyle = '--'
            else:
                color = 'red'
                linestyle = '--'

            axes.plot(curve.flatten(), color=color, linestyle=linestyle, linewidth=2)




            # axes[0].set_title(title, fontsize=10)
            axes.set_xlabel('time')
            axes.set_ylabel('value')
            axes.grid(True, alpha=0.3)


        # for ax in axes[len(indices):]:
        #     ax.axis('off')

        plt.tight_layout()
        plt.savefig('./images/rain_samples.png')

    def plot_prediction_curves(self,
                               X: np.ndarray,
                               predictions: np.ndarray,
                               probabilities: np.ndarray,
                               figsize: Tuple[int, int] = (15, 8),
                               save_path: str = './images/prediction_curves.png'):

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        

        positive_indices = np.where(predictions == 1)[0]
        negative_indices = np.where(predictions == 0)[0]
        

        for idx in negative_indices:
            curve = X[idx].flatten()
            ax.plot(curve, color='gray', alpha=0.6, linewidth=1.5, label='阴性' if idx == negative_indices[0] else '')
        

        for idx in positive_indices:
            curve = X[idx].flatten()
            ax.plot(curve, color='red', alpha=0.7, linewidth=1.5, label='阳性' if idx == positive_indices[0] else '')
        

        ax.set_xlabel('time', fontsize=12)
        ax.set_ylabel('value', fontsize=12)
        ax.set_title(f'predict result\nsample number: {len(X)}, positive: {len(positive_indices)}, negative: {len(negative_indices)}',
                     fontsize=14, fontweight='bold')
        

        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        

        ax.grid(True, alpha=0.3, linestyle='--')
        

        plt.tight_layout()
        

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"The prediction result curve has been saved to: {save_path}")
        

        # plt.show()
        
        plt.close()


def export_training_history(history, filename=None):

    import pandas as pd

    if filename is None:
        filename = "training_history.xlsx"


    history_df = pd.DataFrame(history.history)


    history_df.insert(0, 'epoch', range(len(history_df)))


    history_df.to_excel(filename, index=False)

    print(f"The training history has been saved.: {filename}")
    print(f"Number of training rounds: {len(history_df)}")
    print(f"Incorporating indicators: {list(history_df.columns[1:])}")

    return history_df


def train_new_model(args):

    print("Step 1: Prepare the data")
    path = args.data_path
    X, y = read_many_files(path)
    X = np.array(X)


    print("\nStep 2: Data preprocessing and enhancement")
    processor = CurveDataProcessor()


    X_norm = processor.normalize_curves(X, method='minmax')
    

    use_data_aug = getattr(args, 'use_data_aug', False)
    if use_data_aug:
        print("Apply data augmentation...")
        X_norm, y = processor.augment_time_series(X_norm, np.array(y),
                                                  augment_factor=2,
                                                  balance_classes=True)
        y = np.array(y)
        print(f"Increased data size: {X_norm.shape}")


    print("\nStep 3: Divide the dataset")
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(f"Size of the training set: {X_train.shape}")
    print(f"Size of the test set: {X_test.shape}")
    print(f"Positive ratio - Training set: {np.mean(y_train):.3f}, Test set: {np.mean(y_test):.3f}")


    print("\nStep 4: Create and train the model")

    input_shape = (X_train.shape[1], X_train.shape[2])
    if args.resume is not None and os.path.exists(f"{args.resume}_model.h5"):
        classifier = DeepCurveClassifier(
            input_shape=input_shape,
            model_type=args.model_type,
            random_seed=42
        )
        classifier.load_model(args.resume)
        print(f"*******load {args.resume}")
    else:

        classifier = DeepCurveClassifier(
            input_shape=input_shape,
            model_type=args.model_type,
            random_seed=42
        )


    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train, y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=42
    )
    
    print(f"Final training set size: {X_train_final.shape}")
    print(f"Size of the validation set: {X_val_final.shape}")
    

    history = classifier.train(
        X_train_final, y_train_final,
        X_val=X_val_final,
        y_val=y_val_final,
        args=args,
        use_class_weights=True
    )


    print("\nStep 5: Evaluate the model")

    results = classifier.evaluate(X_test, y_test)

    print("\nEvaluation results:")
    print(f"Accuracy rate: {results['accuracy']:.3f}")
    print(f"ROC AUC: {results['roc_auc']:.3f}")
    print(f"PR AUC: {results['pr_auc']:.3f}")
    print(f"Positive accuracy rate: {results['precision_positive']:.3f}")
    print(f"Positive recall rate: {results['recall_positive']:.3f}")


    print("\nStep 6: Analyze rain points (boundary samples)")


    y_pred, y_proba = classifier.predict(X_test)


    rain_analysis = classifier.analyze_rain_points(
        X_test,
        y_proba,
        uncertainty_threshold=0.2
    )

    print(f"\nRain Point Analysis:")
    print(f"Total sample size: {len(X_test)}")
    print(f"Number of Raindrops: {rain_analysis['rain_count']}")
    print(f"Raindrop proportion: {rain_analysis['rain_percentage']:.2f}%")


    print("\nSome suggestions for Rain points:")
    rain_indices = rain_analysis['rain_indices'][:5]
    for idx in rain_indices:
        print(f"样本 {idx}: {rain_analysis['recommendations'][idx]}")


    print("\nStep 7: Threshold Calibration")


    X_val_cal, X_test_cal, y_val_cal, y_test_cal = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    optimal_threshold = classifier.calibrate_threshold(
        X_val_cal, y_val_cal,
        target_metric='f1'
    )


    print("\nRe-evaluate using the calibration threshold:")
    y_pred_cal = (y_proba >= optimal_threshold).astype(int)
    report = classification_report(y_test, y_pred_cal)
    print(report)


    print("\nStep 8: Visualize the results")


    os.makedirs('./images', exist_ok=True)

    visualizer = CurveVisualizer()


    visualizer.plot_learning_curves(history)


    results['true_labels'] = y_test
    visualizer.plot_classification_results(results)
    

    wrong_indeces = np.where(y_test != y_pred)[0]
    if len(wrong_indeces) > 0:
        x_test_wrong = X_test[wrong_indeces]
        y_test_wrong = np.array(y_test)[wrong_indeces]
        y_pred_wrong = y_pred[wrong_indeces]
        y_proba_wrong = y_proba[wrong_indeces]

        visualizer.plot_sample_curves(
            x_test_wrong,
            y_test_wrong,
            y_pred_wrong,
            y_proba_wrong
        )
    else:
        print("Samples without error classification！")


    print("\nStep 9: Save the model")


    model_save_path = os.path.join(args.work_dir, args.model_name)
    os.makedirs(args.work_dir, exist_ok=True)
    classifier.save_model(model_save_path)
    print("\nStep 10: Export training history to an Excel file")
    export_training_history(classifier.history, f"training_history_{args.model_name}.xlsx")

    return classifier, results, rain_analysis



if __name__ == "__main__":
    args = parse_args()
    
    if args.run_mode == 'train':
        classifier, results, rain_analysis = train_new_model(args)
        
    elif args.run_mode == 'test':

        if not args.resume:
            print("Error: In the test mode, the --resume parameter needs to be specified to load the pre-trained model.")
            exit(1)
        

        model_path = f"{args.resume}_model.h5"
        config_path = f"{args.resume}_config.json"
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print(f"Error: Model file does not exist: {model_path} or {config_path}")
            exit(1)
        

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                input_shape = tuple(config['input_shape'])
                model_type = config.get('model_type', args.model_type)
        except Exception as e:
            print(f"Warning: Unable to read the configuration file. Using default values. Error: {e}")
            input_shape = (50, 1)
            model_type = args.model_type
        

        classifier = DeepCurveClassifier(
            input_shape=input_shape,
            model_type=model_type,
            random_seed=42
        )
        classifier.load_model(args.resume)
        print(f"Model loaded: {args.resume}")
        

        file_path = args.data_path
        if not os.path.exists(file_path):
            print(f"Error: The data path does not exist.: {file_path}")
            exit(1)
            
        prediction_results = classifier.predict_from_file(file_path)

        total_predictions = []
        total_probabilities = []
        for result in prediction_results:
            total_predictions.extend(result['predictions'])
            total_probabilities.extend(result['probabilities'])
        print("\npredict result:")
        print(f"Total sample size: {len(total_predictions)}")
        print(f"Positive quantity: {np.sum(total_predictions)}")
        print(f"Negative quantity: {len(total_predictions) - np.sum(total_predictions)}")
        print(f"Average probability: {np.mean(total_probabilities):.3f}")
        

        print("\nThe prediction result curve is currently being drawn...")
        visualizer = CurveVisualizer()


        if prediction_results:

            all_X_norm = []
            all_predictions = []
            all_probabilities = []

            for result in prediction_results:
                all_X_norm.extend(result['X_normalized'])
                all_predictions.extend(result['predictions'])
                all_probabilities.extend(result['probabilities'])

            all_X_norm = np.array(all_X_norm)
            all_predictions = np.array(all_predictions)
            all_probabilities = np.array(all_probabilities)


            os.makedirs('./images', exist_ok=True)
            save_path = os.path.join('./images', 'prediction_results.png')

            visualizer.plot_prediction_curves(
                X=all_X_norm,
                predictions=all_predictions,
                probabilities=all_probabilities,
                figsize=(15, 8),
                save_path=save_path
            )


            print("\nDetailed forecast information (first 10 samples):")
            print("-" * 80)
            print(f"{'File name':<30} {'Original type':<10} {'Prediction type':<10} {'Prediction probability':<10}")
            print("-" * 80)
        
    elif args.run_mode == 'train_now':

        directory_path = args.data_path
        if not os.path.exists(directory_path):
            print(f"Error: The data path does not exist.: {directory_path}")
            exit(1)
        

        print("Read the data to determine the input shape...")
        X, y = read_many_files(directory_path)
        X = np.array(X)
        
        if len(X) == 0:
            print("Error: Data file not found")
            exit(1)
        
        input_shape = (X.shape[1], X.shape[2])
        print(f"Detected input shape: {input_shape}")
        

        classifier = DeepCurveClassifier(
            input_shape=input_shape,
            model_type=args.model_type,
            random_seed=42
        )
        

        training_history = classifier.train_from_directory(directory_path, args)
        print("Training completed!")
        

        model_save_path = os.path.join(args.work_dir, args.model_name)
        os.makedirs(args.work_dir, exist_ok=True)
        classifier.save_model(model_save_path)
        print(f"The model has been saved to: {model_save_path}")
        
    else:
        print(f"Error: Unsupported operation mode: {args.run_mode}")
        print("Supported mode: train, test, train_now")
        exit(1)
