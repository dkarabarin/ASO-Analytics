"""
Модуль ML моделирования
Шаг 4: Подготовка данных, оптимизация, обучение моделей, важность признаков
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import json


class MLModeler:
    """Класс для ML моделирования"""
    
    def __init__(self, data_path: Path, df: pd.DataFrame = None):
        """
        Args:
            data_path: Путь к папке с данными
            df: DataFrame с данными (если None, загружается из data_with_clusters.pkl)
        """
        self.data_path = Path(data_path)
        self.plots_path = self.data_path / "plots"
        self.plots_path.mkdir(exist_ok=True)
        self.models_path = self.data_path / "saved_models"
        self.models_path.mkdir(exist_ok=True)
        
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_pickle(self.data_path / "data_with_clusters.pkl")
        
        self.results = {}
        self.best_model = None
        self.scaler_X = None
        self.scaler_y = None
        self.le_keyword = None
        self.le_cluster = None
        
        # Признаки БЕЗ информационной утечки
        self.feature_cols = [
            # Текущие значения (доступны в момент прогноза)
            'motiv', 'position', 'organic_us', 'activation_us',
            
            # Лаги (прошлые значения)
            'motiv_lag_7', 'position_lag_7',
            'motiv_lag_14', 'position_lag_14',
            
            # Скользящие средние (прошлые значения)
            'motiv_ma_7', 'position_ma_7', 'motiv_std_7',
            'motiv_ma_14', 'position_ma_14',
            
            # Временные признаки
            'month', 'quarter', 'day_of_week', 'is_weekend',
            
            # Категориальные
            'keyword_encoded', 'cluster_encoded'
        ]
        
        # Исключённые признаки (информационная утечка)
        self.excluded_features = [
            'position_change', 'position_pct_change', 
            'motiv_change', 'motiv_pct_change'
        ]
        
        self.target_col = 'position_delta_7d'
    
    def run_ml_pipeline(self) -> dict:
        """Запуск полного пайплайна ML"""
        print("\n" + "="*80)
        print("ШАГ 4: ML МОДЕЛИРОВАНИЕ")
        print("="*80)
        
        df = self.df
        
        print(f"\nРазмер данных: {len(df):,} записей")
        print(f"Уникальных ключей: {df['keyword'].nunique()}")
        print(f"Количество кластеров: {df['cluster'].nunique()}")
        
        print("\n⚠️ ВАЖНО: Исключены признаки с информационной утечкой:")
        for feat in self.excluded_features:
            print(f"  - {feat} (требует знания будущего)")
        
        print("\nЦЕЛЕВАЯ ПЕРЕМЕННАЯ:")
        print(f"  - {self.target_col} = изменение позиции через 7 дней")
        print("  - Отрицательное значение = улучшение (подъём в топ)")
        
        # 1. Подготовка данных
        X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = self._prepare_data()
        
        # 2. Базовые модели
        self._train_baseline_models(X_train_scaled, X_val_scaled, X_test_scaled, 
                                    y_train_scaled, y_val_scaled, y_test_scaled)
        
        # 3. XGBoost без регуляризации
        self._train_xgboost_no_reg(X_train_scaled, X_val_scaled, X_test_scaled,
                                   y_train_scaled, y_val_scaled, y_test_scaled)
        
        # 4. XGBoost с регуляризацией
        self._train_xgboost_reg(X_train_scaled, X_val_scaled, X_test_scaled,
                                y_train_scaled, y_val_scaled, y_test_scaled)
        
        # 5. LightGBM с регуляризацией
        self._train_lightgbm_reg(X_train_scaled, X_val_scaled, X_test_scaled,
                                 y_train_scaled, y_val_scaled, y_test_scaled)
        
        # 6. Сводная таблица
        self._create_results_table()
        
        # 7. Важность признаков
        self._analyze_feature_importance()
        
        # 8. Визуализация
        self._generate_plots(X_test_scaled, y_test_scaled)
        
        # 9. Сохранение
        self._save_results()
        
        # 10. Выводы
        self._generate_insights()
        
        print("\n" + "="*80)
        print("✅ ШАГ 4 ЗАВЕРШЕН УСПЕШНО!")
        print("="*80)
        
        return self.results
    
    def _prepare_data(self):
        """Подготовка данных для ML"""
        print("\n" + "="*60)
        print("1. ПОДГОТОВКА ДАННЫХ ДЛЯ ML (БЕЗ УТЕЧКИ)")
        print("="*60)
        
        df = self.df
        
        # Кодирование категориальных признаков
        self.le_keyword = LabelEncoder()
        df['keyword_encoded'] = self.le_keyword.fit_transform(df['keyword'])
        
        self.le_cluster = LabelEncoder()
        df['cluster_encoded'] = self.le_cluster.fit_transform(df['cluster'].astype(str))
        
        # Проверка наличия колонок
        available_features = [col for col in self.feature_cols if col in df.columns]
        print(f"  - Доступных признаков: {len(available_features)}")
        
        # Создаём недостающие признаки
        for lag in [14]:
            if f'motiv_lag_{lag}' not in df.columns:
                df[f'motiv_lag_{lag}'] = df.groupby('keyword')['motiv'].shift(lag).fillna(0)
            if f'position_lag_{lag}' not in df.columns:
                df[f'position_lag_{lag}'] = df.groupby('keyword')['position'].shift(lag).fillna(-1)
        
        for window in [14]:
            if f'motiv_ma_{window}' not in df.columns:
                df[f'motiv_ma_{window}'] = df.groupby('keyword')['motiv'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()).fillna(0)
            if f'position_ma_{window}' not in df.columns:
                df[f'position_ma_{window}'] = df.groupby('keyword')['position'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()).fillna(-1)
        
        print(f"  - Признаков (без утечки): {len(self.feature_cols)}")
        print(f"  - Исключены: {', '.join(self.excluded_features)}")
        
        # Очистка данных
        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()
        
        # Замена inf и NaN
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # Удаляем строки с NaN в целевой
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        print(f"  - Данных для обучения: {len(X):,}")
        
        # Разделение на выборки
        print("\n" + "="*60)
        print("2. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")
        print("="*60)
        
        # Сортировка по времени
        df_ml_sorted = df.loc[mask].sort_values('date').reset_index(drop=True)
        X_sorted = X.iloc[df_ml_sorted.index]
        y_sorted = y.iloc[df_ml_sorted.index]
        
        # Разделение: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X_sorted[:train_size]
        y_train = y_sorted[:train_size]
        X_val = X_sorted[train_size:train_size + val_size]
        y_val = y_sorted[train_size:train_size + val_size]
        X_test = X_sorted[train_size + val_size:]
        y_test = y_sorted[train_size + val_size:]
        
        print(f"  - Train: {len(X_train):,} записей ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  - Val: {len(X_val):,} записей ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  - Test: {len(X_test):,} записей ({len(X_test)/len(X)*100:.1f}%)")
        
        # Стандартизация
        print("\n" + "="*60)
        print("3. СТАНДАРТИЗАЦИЯ ДАННЫХ")
        print("="*60)
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
        
        print("  + Стандартизация выполнена")
        
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train_scaled = y_train_scaled
        self.y_val_scaled = y_val_scaled
        self.y_test_scaled = y_test_scaled
        self.y_test_original = y_test.values
        self.feature_names = self.feature_cols
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
    
    def _train_baseline_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Обучение базовых моделей"""
        print("\n" + "="*60)
        print("4. БАЗОВЫЕ МОДЕЛИ ДЛЯ СРАВНЕНИЯ")
        print("="*60)
        
        models = {}
        results = []
        
        # 4.1. Linear Regression
        print("\n4.1. Linear Regression")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  - MAE: {mae:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - R²: {r2:.4f}")
        
        results.append({'Model': 'Linear Regression', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        models['Linear Regression'] = lr
        
        # 4.2. Random Forest
        print("\n4.2. Random Forest")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  - MAE: {mae:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - R²: {r2:.4f}")
        
        results.append({'Model': 'Random Forest', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        models['Random Forest'] = rf
        
        self.baseline_results = results
        self.baseline_models = models
    
    def _train_xgboost_no_reg(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """XGBoost без регуляризации"""
        print("\n" + "="*60)
        print("5. XGBOOST БЕЗ РЕГУЛЯРИЗАЦИИ (ОРИГИНАЛЬНАЯ)")
        print("="*60)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0)
            }
            
            model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            return mean_absolute_error(y_val, y_pred)
        
        print("\n  + Запуск оптимизации (20 итераций)...")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=20, show_progress_bar=True)
        
        print(f"\n  + Лучший MAE на валидации: {study.best_value:.4f}")
        
        # Обучаем финальную модель
        model = xgb.XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train, verbose=False)
        
        # Оценка на тесте
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Оценка на train
        y_pred_train = model.predict(X_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        
        print(f"\n  Результаты на тесте:")
        print(f"    - MAE: {mae:.4f}")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - R²: {r2:.4f}")
        print(f"\n  Проверка переобучения:")
        print(f"    - Train MAE: {mae_train:.4f}")
        print(f"    - Test MAE: {mae:.4f}")
        print(f"    - Разница: {mae_train - mae:.4f}")
        
        self.results['XGBoost_no_reg'] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mae_train': mae_train,
            'params': study.best_params
        }
        self.xgb_no_reg = model
    
    def _train_xgboost_reg(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """XGBoost с регуляризацией"""
        print("\n" + "="*60)
        print("6. XGBOOST С УСИЛЕННОЙ РЕГУЛЯРИЗАЦИЕЙ")
        print("="*60)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.8),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.1, 10.0, log=True),
            }
            
            model = xgb.XGBRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=50,
                eval_metric='mae'
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            return mean_absolute_error(y_val, y_pred)
        
        print("\n  + Запуск оптимизации с регуляризацией (30 итераций)...")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=30, show_progress_bar=True)
        
        print(f"\n  + Лучший MAE на валидации: {study.best_value:.4f}")
        print("  + Лучшие параметры (с регуляризацией):")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")
        
        # Обучаем финальную модель
        model = xgb.XGBRegressor(
            **study.best_params,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='mae'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Оценка на тесте
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Оценка на train
        y_pred_train = model.predict(X_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        
        print(f"\n  Результаты на тесте:")
        print(f"    - MAE: {mae:.4f}")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - R²: {r2:.4f}")
        print(f"\n  Проверка переобучения:")
        print(f"    - Train MAE: {mae_train:.4f}")
        print(f"    - Test MAE: {mae:.4f}")
        print(f"    - Разница: {mae_train - mae:.4f}")
        
        self.results['XGBoost_reg'] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mae_train': mae_train,
            'params': study.best_params
        }
        self.best_model = model
    
    def _train_lightgbm_reg(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """LightGBM с регуляризацией"""
        print("\n" + "="*60)
        print("7. LIGHTGBM С РЕГУЛЯРИЗАЦИЕЙ")
        print("="*60)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.1, 1.0, log=True),
            }
            
            model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val)
            return mean_absolute_error(y_val, y_pred)
        
        print("\n  + Запуск оптимизации LightGBM (20 итераций)...")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=20, show_progress_bar=True)
        
        print(f"\n  + Лучший MAE на валидации: {study.best_value:.4f}")
        
        # Обучаем финальную модель
        model = lgb.LGBMRegressor(**study.best_params, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Оценка на тесте
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n  Результаты на тесте:")
        print(f"    - MAE: {mae:.4f}")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - R²: {r2:.4f}")
        
        self.results['LightGBM_reg'] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'params': study.best_params
        }
        self.lgb_reg = model
    
    def _create_results_table(self):
        """Создание сводной таблицы результатов"""
        print("\n" + "="*60)
        print("8. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("="*60)
        
        results_data = []
        
        # Базовые модели
        for res in self.baseline_results:
            results_data.append(res)
        
        # XGBoost без регуляризации
        if 'XGBoost_no_reg' in self.results:
            r = self.results['XGBoost_no_reg']
            results_data.append({
                'Model': 'XGBoost (без регул.)',
                'MAE': r['mae'],
                'RMSE': r['rmse'],
                'R2': r['r2']
            })
        
        # XGBoost с регуляризацией
        if 'XGBoost_reg' in self.results:
            r = self.results['XGBoost_reg']
            results_data.append({
                'Model': 'XGBoost (с регул.)',
                'MAE': r['mae'],
                'RMSE': r['rmse'],
                'R2': r['r2']
            })
        
        # LightGBM с регуляризацией
        if 'LightGBM_reg' in self.results:
            r = self.results['LightGBM_reg']
            results_data.append({
                'Model': 'LightGBM (с регул.)',
                'MAE': r['mae'],
                'RMSE': r['rmse'],
                'R2': r['r2']
            })
        
        self.results_table = pd.DataFrame(results_data).round(4)
        
        print("\nРезультаты:")
        print(self.results_table.to_string(index=False))
        
        # Лучшая модель
        best_idx = self.results_table['MAE'].idxmin()
        best_model_name = self.results_table.loc[best_idx, 'Model']
        print(f"\n  + Лучшая модель: {best_model_name}")
        print(f"    - MAE: {self.results_table.loc[best_idx, 'MAE']:.4f}")
        print(f"    - RMSE: {self.results_table.loc[best_idx, 'RMSE']:.4f}")
        print(f"    - R2: {self.results_table.loc[best_idx, 'R2']:.4f}")
        
        self.best_model_name = best_model_name
    
    def _analyze_feature_importance(self):
        """Анализ важности признаков"""
        print("\n" + "="*60)
        print("9. ВАЖНОСТЬ ПРИЗНАКОВ")
        print("="*60)
        
        # Группы признаков для цветов
        self.feature_groups = {
            'Позиция': {
                'features': ['position', 'position_lag_7', 'position_lag_14', 'position_ma_7', 'position_ma_14'],
                'color': '#FF6B6B'
            },
            'Мотив': {
                'features': ['motiv', 'motiv_lag_7', 'motiv_lag_14', 'motiv_ma_7', 'motiv_ma_14', 'motiv_std_7'],
                'color': '#4ECDC4'
            },
            'Временные': {
                'features': ['month', 'quarter', 'day_of_week', 'is_weekend'],
                'color': '#45B7D1'
            },
            'Органика/Активации': {
                'features': ['organic_us', 'activation_us'],
                'color': '#96CEB4'
            },
            'Кластер/Ключ': {
                'features': ['cluster_encoded', 'keyword_encoded'],
                'color': '#FFEAA7'
            }
        }
        
        def get_feature_group(feature_name):
            for group_name, group_info in self.feature_groups.items():
                if feature_name in group_info['features']:
                    return group_name
            return 'Другие'
        
        def get_feature_color(feature_name):
            for group_name, group_info in self.feature_groups.items():
                if feature_name in group_info['features']:
                    return group_info['color']
            return '#D3D3D3'
        
        # Важность для лучшей модели (XGBoost с регуляризацией)
        if self.best_model is not None and hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            imp_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance,
                'importance_pct': importance * 100,
                'group': [get_feature_group(f) for f in self.feature_names],
                'color': [get_feature_color(f) for f in self.feature_names]
            }).sort_values('importance', ascending=False)
            
            print("\nТоп-15 важнейших признаков:")
            print("  {:<25s} {:>12s} {:>12s} {:>20s}".format('Признак', 'Важность', 'Важность %', 'Группа'))
            print("  " + "-"*70)
            
            for i, row in imp_df.head(15).iterrows():
                print("  {:<25s} {:>12.4f} {:>11.2f}% {:>20s}".format(
                    row['feature'][:25],
                    row['importance'],
                    row['importance_pct'],
                    row['group']
                ))
            
            # Суммарная важность по группам
            group_importance = imp_df.groupby('group')['importance'].sum().sort_values(ascending=False)
            
            print("\nГруппы признаков:")
            for group, imp in group_importance.items():
                pct = (imp / group_importance.sum()) * 100
                print(f"   - {group}: {pct:.1f}%")
            
            self.imp_df = imp_df
            self.group_importance = group_importance
    
    def _generate_plots(self, X_test, y_test):
        """Генерация графиков"""
        print("\n" + "="*60)
        print("10. ВИЗУАЛИЗАЦИЯ")
        print("="*60)
        
        # 1. Сравнение моделей
        self._plot_model_comparison()
        
        # 2. Важность признаков
        self._plot_feature_importance()
        
        # 3. Факт vs прогноз
        self._plot_predictions_vs_actual(X_test, y_test)
    
    def _plot_model_comparison(self):
        """График сравнения моделей"""
        print("\n  + Построение графика сравнения моделей...")
        
        names = self.results_table['Model'].tolist()
        mae_vals = self.results_table['MAE'].tolist()
        rmse_vals = self.results_table['RMSE'].tolist()
        r2_vals = self.results_table['R2'].tolist()
        
        # Цвета для разных типов моделей
        colors = []
        for name in names:
            if 'Linear' in name or 'Random' in name:
                colors.append('#B0B0B0')
            elif 'без регул' in name:
                colors.append('#FF6B6B')
            else:
                colors.append('#4ECDC4')
        
        x = np.arange(len(names))
        width = 0.25
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.suptitle('Сравнение моделей: с регуляризацией и без', fontsize=16, fontweight='bold')
        
        # MAE
        ax = axes[0]
        bars = ax.bar(x, mae_vals, width, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Модель', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title('MAE (меньше = лучше)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, mae_vals)):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        best_idx = np.argmin(mae_vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # RMSE
        ax = axes[1]
        bars = ax.bar(x, rmse_vals, width, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Модель', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('RMSE (меньше = лучше)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, rmse_vals)):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        best_idx = np.argmin(rmse_vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # R2
        ax = axes[2]
        bars = ax.bar(x, r2_vals, width, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Модель', fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title('R² (больше = лучше)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, r2_vals)):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        best_idx = np.argmax(r2_vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # Легенда
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, facecolor='#B0B0B0', label='Базовые модели'),
            plt.Rectangle((0,0), 1, 1, facecolor='#FF6B6B', label='Без регуляризации'),
            plt.Rectangle((0,0), 1, 1, facecolor='#4ECDC4', label='С регуляризацией')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'model_comparison_with_regularization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  + График сравнения моделей сохранён: {self.plots_path / 'model_comparison_with_regularization.png'}")
    
    def _plot_feature_importance(self):
        """График важности признаков"""
        print("\n  + Построение графика важности признаков...")
        
        if not hasattr(self, 'imp_df'):
            print("  ⚠️ Данные о важности признаков отсутствуют")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        fig.suptitle('Важность признаков (XGBoost с регуляризацией)', fontsize=16, fontweight='bold')
        
        # График 1: Топ-20 признаков
        ax = axes[0]
        imp_top20 = self.imp_df.head(20)
        bar_colors = imp_top20['color'].tolist()
        
        bars = ax.barh(imp_top20['feature'], imp_top20['importance'], 
                       color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Важность', fontsize=12)
        ax.set_title('Топ-20 важнейших признаков', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (idx, row) in enumerate(imp_top20.iterrows()):
            ax.text(row['importance'] + 0.002, i, f'{row["importance"]:.3f}',
                    va='center', fontsize=8)
        
        # График 2: Суммарная важность по группам
        ax = axes[1]
        group_importance = self.group_importance
        
        group_colors = {
            'Позиция': '#FF6B6B',
            'Мотив': '#4ECDC4',
            'Временные': '#45B7D1',
            'Органика/Активации': '#96CEB4',
            'Кластер/Ключ': '#FFEAA7'
        }
        
        colors_group = [group_colors.get(g, '#D3D3D3') for g in group_importance.index]
        
        bars = ax.barh(group_importance.index, group_importance.values, 
                       color=colors_group, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Суммарная важность', fontsize=12)
        ax.set_title('Важность по группам признаков', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        total_importance = group_importance.sum()
        for i, (idx, val) in enumerate(group_importance.items()):
            pct = (val / total_importance) * 100
            ax.text(val + 0.005, i, f'{val:.4f} ({pct:.1f}%)',
                    va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'feature_importance_by_group.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  + График важности признаков сохранён: {self.plots_path / 'feature_importance_by_group.png'}")
    
    def _plot_predictions_vs_actual(self, X_test, y_test):
        """График факт vs прогноз"""
        print("\n  + Построение графика факт vs прогноз...")
        
        if self.best_model is None:
            print("  ⚠️ Лучшая модель отсутствует")
            return
        
        # Предсказания лучшей модели
        y_pred_scaled = self.best_model.predict(X_test)
        y_test_original = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_original = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(y_test_original, y_pred_original, alpha=0.4, s=15, color='steelblue')
        ax.plot([y_test_original.min(), y_test_original.max()],
                [y_test_original.min(), y_test_original.max()],
                'r--', linewidth=2, label='Идеальное предсказание')
        ax.set_xlabel('Фактическое изменение позиции', fontsize=12)
        ax.set_ylabel('Предсказанное изменение позиции', fontsize=12)
        ax.set_title(f'Факт vs Прогноз (XGBoost с регуляризацией)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Добавляем метрики
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        
        ax.text(0.05, 0.95, f'MAE: {mae:.4f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
        ax.text(0.05, 0.90, f'R²: {r2:.4f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'predictions_vs_actual.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  + График факт vs прогноз сохранён: {self.plots_path / 'predictions_vs_actual.png'}")
    
    def _save_results(self):
        """Сохранение результатов"""
        print("\n" + "="*60)
        print("11. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*60)
        
        # Результаты моделей
        self.results_table.to_csv(self.data_path / "model_results_final.csv", index=False)
        print(f"  + Результаты сохранены: {self.data_path / 'model_results_final.csv'}")
        
        # Лучшая модель
        if self.best_model is not None:
            joblib.dump(self.best_model, self.models_path / "best_model_xgboost_final.pkl")
            print(f"  + Лучшая модель сохранена: {self.models_path / 'best_model_xgboost_final.pkl'}")
        
        # Scaler
        if self.scaler_X is not None:
            joblib.dump(self.scaler_X, self.models_path / "scaler_final.pkl")
            print(f"  + Scaler сохранён: {self.models_path / 'scaler_final.pkl'}")
        
        # Важность признаков
        if hasattr(self, 'imp_df'):
            self.imp_df.to_csv(self.data_path / "feature_importance_final.csv", index=False)
            print(f"  + Важность признаков сохранена: {self.data_path / 'feature_importance_final.csv'}")
        
        # Кодировщики
        if self.le_keyword is not None:
            joblib.dump(self.le_keyword, self.models_path / "le_keyword.pkl")
        if self.le_cluster is not None:
            joblib.dump(self.le_cluster, self.models_path / "le_cluster.pkl")
        print(f"  + Кодировщики сохранены")
    
    def _generate_insights(self):
        """Генерация выводов"""
        print("\n" + "="*60)
        print("12. ИТОГОВЫЕ ВЫВОДЫ")
        print("="*60)
        
        if 'XGBoost_reg' in self.results:
            xgb_reg = self.results['XGBoost_reg']
            xgb_no_reg = self.results.get('XGBoost_no_reg', {})
            
            print(f"""
📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:

1. ЛУЧШАЯ МОДЕЛЬ: {self.best_model_name}
   - MAE:  {xgb_reg['mae']:.4f}
   - RMSE: {xgb_reg['rmse']:.4f}
   - R²:   {xgb_reg['r2']:.4f}

2. СРАВНЕНИЕ С РЕГУЛЯРИЗАЦИЕЙ:
   - Без регуляризации: MAE = {xgb_no_reg.get('mae', 0):.4f}, R² = {xgb_no_reg.get('r2', 0):.4f}
   - С регуляризацией:  MAE = {xgb_reg['mae']:.4f}, R² = {xgb_reg['r2']:.4f}
   - Улучшение: MAE улучшился на {xgb_no_reg.get('mae', 0) - xgb_reg['mae']:.4f}

3. ПЕРЕОБУЧЕНИЕ:
   - Без регуляризации: разница Train-Test = {xgb_no_reg.get('mae_train', 0) - xgb_no_reg.get('mae', 0):.4f}
   - С регуляризацией:  разница Train-Test = {xgb_reg.get('mae_train', 0) - xgb_reg['mae']:.4f}
   - {'✅ Регуляризация уменьшила переобучение' if xgb_no_reg.get('mae_train', 0) - xgb_no_reg.get('mae', 0) > xgb_reg.get('mae_train', 0) - xgb_reg['mae'] else '⚠️ Регуляризация не помогла'}

4. ВАЖНЕЙШИЕ ПРИЗНАКИ:
""")
            
            if hasattr(self, 'imp_df'):
                for i, row in self.imp_df.head(5).iterrows():
                    print(f"   {i+1}. {row['feature']}: {row['importance_pct']:.2f}%")
            
            print("""
5. РЕКОМЕНДАЦИИ:
   - Используйте XGBoost С РЕГУЛЯРИЗАЦИЕЙ для продакшена
   - Модель меньше переобучается и даёт более стабильные прогнозы
   - Основные драйверы изменения позиции: текущая позиция и её лаги
   - Мотив важен, но не является главным фактором
""")
    
    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """Предсказание для новых данных"""
        if self.best_model is None:
            raise ValueError("Модель не обучена. Сначала запустите run_ml_pipeline()")
        
        # Проверка наличия всех признаков
        missing = set(self.feature_cols) - set(X_new.columns)
        if missing:
            raise ValueError(f"Отсутствуют признаки: {missing}")
        
        # Стандартизация
        X_scaled = self.scaler_X.transform(X_new[self.feature_cols])
        
        # Предсказание
        y_pred_scaled = self.best_model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return y_pred
    
    def load_model(self):
        """Загрузка сохранённой модели"""
        model_path = self.models_path / "best_model_xgboost_final.pkl"
        scaler_path = self.models_path / "scaler_final.pkl"
        
        if model_path.exists() and scaler_path.exists():
            self.best_model = joblib.load(model_path)
            self.scaler_X = joblib.load(scaler_path)
            print(f"✅ Модель загружена из {model_path}")
            return True
        else:
            print(f"⚠️ Модель не найдена")
            return False