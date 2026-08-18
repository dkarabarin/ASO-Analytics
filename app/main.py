"""
FastAPI приложение для ASO Analytics Service
Главный файл с эндпоинтами и маршрутами
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import json
import os
import logging
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорт модулей
from app.core.data_loader import DataLoader
from app.core.eda import EDAAnalyzer
from app.core.clustering import ClusterAnalyzer
from app.core.ml_model import MLModeler
from app.core.statistical_tests import StatisticalTester
from app.core.reporting import ReportGenerator
from app.llm.client import LLMClient

# Импорт схем
from app.api.schemas import (
    StatusResponse,
    ErrorResponse,
    SuccessResponse,
    DataLoadResponse,
    DataStatusResponse,
    EDASummaryResponse,
    TopKeywordsResponse,
    PlotsListResponse,
    PlotsByStepResponse,
    PlotsGroupedResponse,
    PlotItem,
    ClusteringRunResponse,
    ClustersResponse,
    ClusterInfo,
    AnchorWordsResponse,
    AnchorWord,
    MLTrainResponse,
    MLResultsResponse,
    ModelResult,
    FeatureImportanceResponse,
    FeatureImportanceItem,
    PredictRequest,
    PredictResponse,
    TestsResultsResponse,
    ReportGenerateResponse,
    ReportResponse,
    LLMAnalyzeRequest,
    LLMChatRequest,
    LLMResponse,
    LLMStatusResponse,
    MethodologyResponse,
    FilterParams,
    ExportRequest,
    ExplainPlotRequest
)

# =====================================================
# КОНФИГУРАЦИЯ
# =====================================================

DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)
PLOTS_PATH = DATA_PATH / "plots"
PLOTS_PATH.mkdir(exist_ok=True)
MODELS_PATH = DATA_PATH / "saved_models"
MODELS_PATH.mkdir(exist_ok=True)
STATIC_PATH = Path("app/static")
STATIC_PATH.mkdir(parents=True, exist_ok=True)

# =====================================================
# СОЗДАНИЕ ПРИЛОЖЕНИЯ
# =====================================================

app = FastAPI(
    title="ASO Analytics Service",
    description="Сервис для анализа ASO данных приложения Sound Amplifier",
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статика
app.mount("/plots", StaticFiles(directory=str(PLOTS_PATH)), name="plots")
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

# =====================================================
# СОСТОЯНИЕ ПРИЛОЖЕНИЯ
# =====================================================

class AppState:
    """Состояние приложения"""
    def __init__(self):
        self.df = None
        self.data_loaded = False
        self.eda_completed = False
        self.eda_results = None
        self.eda_analyzer = None
        self.clustering_completed = False
        self.clusterer = None
        self.ml_completed = False
        self.ml_model = None
        self.ml_results = None
        self.tests_completed = False
        self.tester = None
        self.report_generated = False
        self.current_report = None
        self.llm_client = LLMClient()

state = AppState()

# =====================================================
# ФУНКЦИЯ ЗАГРУЗКИ СУЩЕСТВУЮЩИХ РЕЗУЛЬТАТОВ
# =====================================================

def load_existing_results():
    """Загрузка существующих результатов из файлов"""
    try:
        # Проверяем данные
        if (DATA_PATH / "clean_data.pkl").exists():
            state.df = pd.read_pickle(DATA_PATH / "clean_data.pkl")
            state.data_loaded = True
            logger.info(f"✅ Загружены данные: {len(state.df)} записей")
            
            # Проверяем кластеры
            if 'cluster' in state.df.columns:
                state.clustering_completed = True
                logger.info(f"✅ Загружены кластеры: {state.df['cluster'].nunique()}")
            
            # Проверяем EDA
            if (DATA_PATH / "plots").exists() and len(list((DATA_PATH / "plots").glob("*.png"))) > 0:
                state.eda_completed = True
                logger.info(f"✅ Загружены графики EDA: {len(list((DATA_PATH / 'plots').glob('*.png')))} файлов")
            
            # Проверяем ML
            if (DATA_PATH / "model_results_final.csv").exists():
                state.ml_completed = True
                logger.info(f"✅ Загружены результаты ML")
            
            # Проверяем тесты
            if (DATA_PATH / "hypotheses_by_cluster_summary.csv").exists():
                state.tests_completed = True
                logger.info(f"✅ Загружены результаты тестов")
            
            # Проверяем отчёт
            if (DATA_PATH / "final_report.json").exists():
                state.report_generated = True
                logger.info(f"✅ Загружен отчёт")
                
    except Exception as e:
        logger.warning(f"⚠️ Ошибка загрузки существующих результатов: {e}")

# =====================================================
# ЗАПУСК ПРИ СТАРТЕ
# =====================================================

@app.on_event("startup")
async def startup_event():
    load_existing_results()

# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================

def get_data_summary() -> dict:
    """Получить сводку по данным"""
    if state.df is None:
        return {}
    
    return {
        'total_records': len(state.df),
        'unique_keywords': state.df['keyword'].nunique(),
        'unique_clusters': state.df['cluster'].nunique() if 'cluster' in state.df.columns else 0,
        'period_start': state.df['date'].min().strftime('%Y-%m-%d'),
        'period_end': state.df['date'].max().strftime('%Y-%m-%d'),
        'avg_motiv': float(state.df['motiv'].mean()),
        'avg_position': float(state.df[state.df['position'] != -1]['position'].mean()) 
                        if len(state.df[state.df['position'] != -1]) > 0 else 0
    }

def clean_float_values(obj):
    """Очистка float значений от NaN и Inf для JSON"""
    if isinstance(obj, dict):
        return {k: clean_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_float_values(v) for v in obj]
    elif isinstance(obj, float):
        if pd.isna(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj

# =====================================================
# ЭНДПОИНТЫ: ЗАГРУЗКА ДАННЫХ (Шаг 1)
# =====================================================

@app.post(
    "/api/data/load",
    response_model=DataLoadResponse,
    summary="Загрузка и предобработка данных"
)
async def load_data():
    """
    Загрузка данных из Excel файлов:
    - Sound Amplifier_2024.xlsx
    - Sound Amplifier_2025.xlsx
    - Sound Amplifier_2026.xlsx
    """
    try:
        logger.info("Начало загрузки данных...")
        loader = DataLoader(DATA_PATH)
        state.df = loader.load_and_preprocess()
        state.data_loaded = True
        logger.info(f"Данные загружены: {len(state.df)} записей")
        
        return DataLoadResponse(
            status="success",
            message="Данные успешно загружены и обработаны",
            records=len(state.df),
            keywords=state.df['keyword'].nunique(),
            period_start=state.df['date'].min().strftime('%Y-%m-%d'),
            period_end=state.df['date'].max().strftime('%Y-%m-%d')
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/data/status",
    response_model=DataStatusResponse,
    summary="Статус загрузки данных"
)
async def get_data_status():
    """Проверка статуса загрузки данных"""
    if not state.data_loaded or state.df is None:
        return DataStatusResponse(loaded=False)
    
    return DataStatusResponse(
        loaded=True,
        records=len(state.df),
        keywords=state.df['keyword'].nunique(),
        clusters=state.df['cluster'].nunique() if 'cluster' in state.df.columns else 0,
        period_start=state.df['date'].min().strftime('%Y-%m-%d'),
        period_end=state.df['date'].max().strftime('%Y-%m-%d')
    )


@app.post("/api/data/reset", response_model=SuccessResponse)
async def reset_data():
    """Сброс всех данных и состояния"""
    state.df = None
    state.data_loaded = False
    state.eda_completed = False
    state.clustering_completed = False
    state.ml_completed = False
    state.tests_completed = False
    state.report_generated = False
    state.current_report = None
    
    return SuccessResponse(
        status="success",
        message="Состояние сброшено",
        data={"reset": True}
    )


# =====================================================
# ЭНДПОИНТЫ: EDA (Шаг 2)
# =====================================================

@app.post(
    "/api/eda/run",
    response_model=SuccessResponse,
    summary="Запуск EDA"
)
async def run_eda():
    """Запуск разведочного анализа данных"""
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    
    try:
        logger.info("Начало EDA...")
        eda = EDAAnalyzer(DATA_PATH, state.df)
        results = eda.run_eda()
        state.eda_completed = True
        state.eda_results = results
        state.eda_analyzer = eda
        
        plots_count = len(list(PLOTS_PATH.glob("*.png")))
        logger.info(f"EDA завершён. Создано графиков: {plots_count}")
        
        return SuccessResponse(
            status="success",
            message="EDA завершён успешно",
            data={
                "plots_count": plots_count,
                "results": {
                    "total_records": results.get('general_stats', {}).get('total_records', 0),
                    "unique_keywords": results.get('general_stats', {}).get('unique_keywords', 0)
                }
            }
        )
    except Exception as e:
        logger.error(f"Ошибка EDA: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка EDA: {str(e)}")


@app.get(
    "/api/eda/summary",
    response_model=Dict[str, Any],
    summary="Сводная статистика EDA"
)
async def get_eda_summary():
    """Получить сводную статистику из EDA"""
    if not state.eda_completed:
        raise HTTPException(status_code=400, detail="EDA не выполнен")
    
    return clean_float_values(state.eda_results.get('general_stats', {}))


@app.get(
    "/api/eda/top_keywords",
    response_model=TopKeywordsResponse,
    summary="Топ-N ключевых слов"
)
async def get_top_keywords(limit: int = Query(20, ge=1, le=100)):
    """Получить топ-N ключевых слов по количеству записей"""
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    
    top = state.df['keyword'].value_counts().head(limit)
    return TopKeywordsResponse(top_keywords=top.to_dict())


@app.get(
    "/api/eda/plots",
    response_model=PlotsListResponse,
    summary="Список доступных графиков"
)
async def get_plots_list():
    """Получить список всех сгенерированных графиков"""
    plots = []
    for f in PLOTS_PATH.glob("*.png"):
        plots.append(PlotItem(
            name=f.name,
            url=f"/plots/{f.name}",
            size=f.stat().st_size,
            modified=f.stat().st_mtime
        ))
    
    return PlotsListResponse(plots=plots)


@app.get(
    "/api/eda/insights",
    response_model=Dict[str, Any],
    summary="Ключевые выводы EDA"
)
async def get_eda_insights():
    """Получить ключевые выводы из EDA"""
    if not state.eda_completed:
        raise HTTPException(status_code=400, detail="EDA не выполнен")
    
    return clean_float_values(state.eda_results.get('insights', {}))


# =====================================================
# ЭНДПОИНТЫ: ГРАФИКИ ПО ШАГАМ
# =====================================================

@app.get("/api/plots/by_step/{step}", response_model=PlotsByStepResponse)
async def get_plots_by_step(step: int):
    """Получить графики для конкретного шага"""
    step_names = {
        1: "Загрузка данных",
        2: "EDA (Разведочный анализ)",
        3: "Кластеризация",
        4: "ML моделирование",
        5: "Статистические тесты",
        6: "Проверка правил"
    }
    
    step_patterns = {
        1: ['clean_data'],
        2: ['eda_overview', 'correlation_heatmap', 'timeseries_top5', 
            'monthly_analysis', 'motiv_distribution', 'position_distribution',
            'top20_keywords', 'motiv_vs_position'],
        3: ['cluster_analysis', 'cluster_timeline'],
        4: ['model_comparison', 'feature_importance', 'predictions_vs_actual'],
        5: ['hypotheses_by_cluster'],
        6: ['rules_from_document']
    }
    
    patterns = step_patterns.get(step, [])
    plots = []
    
    for f in PLOTS_PATH.glob("*.png"):
        for pattern in patterns:
            if pattern in f.name:
                plots.append(PlotItem(
                    name=f.name,
                    url=f"/plots/{f.name}",
                    size=f.stat().st_size,
                    modified=f.stat().st_mtime,
                    step=step,
                    step_name=step_names.get(step, f"Шаг {step}")
                ))
                break
    
    return PlotsByStepResponse(
        step=step,
        step_name=step_names.get(step, f"Шаг {step}"),
        plots=plots,
        total=len(plots)
    )


@app.get("/api/plots/all_grouped", response_model=PlotsGroupedResponse)
async def get_all_plots_grouped():
    """Получить все графики, сгруппированные по шагам"""
    step_names = {
        1: "Загрузка данных",
        2: "EDA (Разведочный анализ)",
        3: "Кластеризация",
        4: "ML моделирование",
        5: "Статистические тесты",
        6: "Проверка правил"
    }
    
    step_patterns = {
        1: ['clean_data'],
        2: ['eda_overview', 'correlation_heatmap', 'timeseries_top5', 
            'monthly_analysis', 'motiv_distribution', 'position_distribution',
            'top20_keywords', 'motiv_vs_position'],
        3: ['cluster_analysis', 'cluster_timeline'],
        4: ['model_comparison', 'feature_importance', 'predictions_vs_actual'],
        5: ['hypotheses_by_cluster'],
        6: ['rules_from_document']
    }
    
    grouped = {}
    for step in range(1, 7):
        grouped[step] = {
            "name": step_names.get(step, f"Шаг {step}"),
            "plots": []
        }
    
    grouped[0] = {"name": "Другие", "plots": []}
    
    for f in PLOTS_PATH.glob("*.png"):
        name = f.name
        assigned = False
        
        for step, patterns in step_patterns.items():
            for pattern in patterns:
                if pattern in name:
                    grouped[step]["plots"].append(PlotItem(
                        name=name,
                        url=f"/plots/{name}",
                        size=f.stat().st_size,
                        modified=f.stat().st_mtime,
                        step=step,
                        step_name=step_names.get(step, f"Шаг {step}")
                    ))
                    assigned = True
                    break
            if assigned:
                break
        
        if not assigned:
            grouped[0]["plots"].append(PlotItem(
                name=name,
                url=f"/plots/{name}",
                size=f.stat().st_size,
                modified=f.stat().st_mtime
            ))
    
    return PlotsGroupedResponse(
        plots_by_step=grouped,
        total=len(list(PLOTS_PATH.glob("*.png")))
    )


@app.get("/api/plots/step/{step}/count")
async def get_plots_count_by_step(step: int):
    """Получить количество графиков для шага"""
    result = await get_plots_by_step(step)
    return {"step": step, "count": result.total}


# =====================================================
# ЭНДПОИНТЫ: КЛАСТЕРИЗАЦИЯ (Шаг 3)
# =====================================================

@app.post(
    "/api/clustering/run",
    response_model=ClusteringRunResponse,
    summary="Запуск кластеризации"
)
async def run_clustering():
    """Запуск бизнес-кластеризации"""
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    
    try:
        logger.info("Начало кластеризации...")
        clusterer = ClusterAnalyzer(DATA_PATH, state.df)
        state.df = clusterer.run_clustering()
        state.clustering_completed = True
        state.clusterer = clusterer
        logger.info(f"Кластеризация завершена: {state.df['cluster'].nunique()} кластеров")
        
        return ClusteringRunResponse(
            status="success",
            message="Кластеризация завершена",
            clusters=state.df['cluster'].nunique()
        )
    except Exception as e:
        logger.error(f"Ошибка кластеризации: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/clustering/clusters",
    response_model=ClustersResponse,
    summary="Информация о кластерах"
)
async def get_clusters():
    """Получить информацию о всех кластерах"""
    if not state.clustering_completed:
        raise HTTPException(status_code=400, detail="Кластеризация не выполнена")
    
    clusters = {}
    for cluster, group in state.df.groupby('cluster'):
        clusters[cluster] = ClusterInfo(
            keywords=group['keyword'].nunique(),
            motiv_mean=float(group['motiv'].mean()) if not pd.isna(group['motiv'].mean()) else 0.0,
            position_mean=float(group[group['position'] != -1]['position'].mean()) if len(group[group['position'] != -1]) > 0 else 0.0,
            motiv_sum=float(group['motiv'].sum()) if not pd.isna(group['motiv'].sum()) else 0.0,
            position_min=float(group[group['position'] != -1]['position'].min()) if len(group[group['position'] != -1]) > 0 else 0.0,
            position_max=float(group[group['position'] != -1]['position'].max()) if len(group[group['position'] != -1]) > 0 else 0.0
        )
    
    return ClustersResponse(clusters=clusters)


@app.get(
    "/api/clustering/anchor_words",
    response_model=AnchorWordsResponse,
    summary="Якорные слова"
)
async def get_anchor_words():
    """Получить якорные слова для всех кластеров"""
    if not state.clustering_completed:
        raise HTTPException(status_code=400, detail="Кластеризация не выполнена")
    
    anchor_path = DATA_PATH / "anchor_words.csv"
    if anchor_path.exists():
        df = pd.read_csv(anchor_path)
        anchor_words = [
            AnchorWord(
                cluster=row['cluster'],
                anchor_word=row['anchor_word'],
                motiv_mean=float(row['motiv_mean']) if not pd.isna(row['motiv_mean']) else 0.0,
                motiv_sum=float(row['motiv_sum']) if not pd.isna(row['motiv_sum']) else 0.0,
                position_mean=float(row['position_mean']) if not pd.isna(row['position_mean']) else 0.0,
                anchor_score=float(row['anchor_score']) if not pd.isna(row['anchor_score']) else 0.0
            )
            for _, row in df.iterrows()
        ]
        return AnchorWordsResponse(anchor_words=anchor_words)
    
    return AnchorWordsResponse(anchor_words=[])


# =====================================================
# ЭНДПОИНТЫ: ML (Шаг 4)
# =====================================================

@app.post(
    "/api/ml/train",
    response_model=MLTrainResponse,
    summary="Обучение ML моделей"
)
async def train_models():
    """Обучение ML моделей"""
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    
    try:
        logger.info("Начало обучения ML моделей...")
        ml = MLModeler(DATA_PATH, state.df)
        results = ml.run_ml_pipeline()
        state.ml_completed = True
        state.ml_model = ml
        state.ml_results = results
        logger.info(f"ML обучение завершено. Лучшая модель: {ml.best_model_name}")
        
        return MLTrainResponse(
            status="success",
            message="Обучение моделей завершено",
            best_model=ml.best_model_name
        )
    except Exception as e:
        logger.error(f"Ошибка обучения ML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/ml/results",
    response_model=MLResultsResponse,
    summary="Результаты ML моделей"
)
async def get_ml_results():
    """Получить результаты всех обученных моделей"""
    if not state.ml_completed:
        raise HTTPException(status_code=400, detail="ML модели не обучены")
    
    results_path = DATA_PATH / "model_results_final.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        results = []
        for _, row in df.iterrows():
            results.append(ModelResult(
                Model=row['Model'],
                MAE=float(row['MAE']) if not pd.isna(row['MAE']) else 0.0,
                RMSE=float(row['RMSE']) if not pd.isna(row['RMSE']) else 0.0,
                R2=float(row['R2']) if not pd.isna(row['R2']) else 0.0
            ))
        return MLResultsResponse(results=results)
    
    return MLResultsResponse(results=[])


@app.get(
    "/api/ml/feature_importance",
    response_model=FeatureImportanceResponse,
    summary="Важность признаков"
)
async def get_feature_importance():
    """Получить важность признаков для лучшей модели"""
    if not state.ml_completed:
        raise HTTPException(status_code=400, detail="ML модели не обучены")
    
    importance_path = DATA_PATH / "feature_importance_final.csv"
    if importance_path.exists():
        df = pd.read_csv(importance_path)
        features = []
        for _, row in df.head(15).iterrows():
            features.append(FeatureImportanceItem(
                feature=row['feature'],
                importance=float(row['importance']) if not pd.isna(row['importance']) else 0.0,
                importance_pct=float(row['importance_pct']) if not pd.isna(row['importance_pct']) else 0.0,
                group=row['group']
            ))
        return FeatureImportanceResponse(feature_importance=features)
    
    return FeatureImportanceResponse(feature_importance=[])


@app.post(
    "/api/ml/predict",
    response_model=PredictResponse,
    summary="Предсказание изменения позиции"
)
async def predict(request: PredictRequest):
    """Предсказать изменение позиции через 7 дней"""
    if not state.ml_completed:
        ml = MLModeler(DATA_PATH, state.df)
        if ml.load_model():
            state.ml_model = ml
            state.ml_completed = True
        else:
            raise HTTPException(status_code=400, detail="Модель не обучена")
    
    try:
        data = request.dict()
        
        if data.get('keyword') and state.ml_model.le_keyword is not None:
            try:
                data['keyword_encoded'] = state.ml_model.le_keyword.transform([data['keyword']])[0]
            except:
                data['keyword_encoded'] = -1
        
        if data.get('cluster') and state.ml_model.le_cluster is not None:
            try:
                data['cluster_encoded'] = state.ml_model.le_cluster.transform([data['cluster']])[0]
            except:
                data['cluster_encoded'] = -1
        
        X_new = pd.DataFrame([data])
        
        for col in state.ml_model.feature_cols:
            if col not in X_new.columns:
                X_new[col] = 0
        
        prediction = state.ml_model.predict(X_new[X_new.columns.intersection(state.ml_model.feature_cols)])
        
        pred_val = float(prediction[0]) if not pd.isna(prediction[0]) else 0.0
        
        if pred_val < 0:
            interpretation = f"✅ Ожидается улучшение позиции на {abs(pred_val):.2f} позиций"
        elif pred_val > 0:
            interpretation = f"⚠️ Ожидается ухудшение позиции на {pred_val:.2f} позиций"
        else:
            interpretation = "➖ Позиция останется без изменений"
        
        return PredictResponse(
            prediction=pred_val,
            prediction_original=pred_val,
            interpretation=interpretation
        )
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/load_model", response_model=SuccessResponse)
async def load_saved_model():
    """Загрузка сохранённой модели из файла"""
    ml = MLModeler(DATA_PATH, state.df)
    if ml.load_model():
        state.ml_model = ml
        state.ml_completed = True
        return SuccessResponse(
            status="success",
            message="Модель загружена успешно"
        )
    else:
        raise HTTPException(status_code=404, detail="Модель не найдена")


# =====================================================
# ЭНДПОИНТЫ: СТАТИСТИЧЕСКИЕ ТЕСТЫ (Шаг 5)
# =====================================================

@app.post(
    "/api/tests/run",
    response_model=SuccessResponse,
    summary="Запуск статистических тестов"
)
async def run_tests():
    """Запуск статистических тестов по кластерам"""
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    
    try:
        logger.info("Начало статистических тестов...")
        tester = StatisticalTester(DATA_PATH, state.df)
        results = tester.run_tests()
        state.tests_completed = True
        state.tester = tester
        logger.info("Статистические тесты завершены")
        
        return SuccessResponse(
            status="success",
            message="Статистические тесты завершены",
            data={"clusters_tested": len(tester.clusters)}
        )
    except Exception as e:
        logger.error(f"Ошибка статистических тестов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/tests/results",
    response_model=TestsResultsResponse,
    summary="Результаты статистических тестов"
)
async def get_tests_results():
    """Получить результаты всех статистических тестов"""
    if not state.tests_completed:
        raise HTTPException(status_code=400, detail="Тесты не выполнены")
    
    hypotheses_path = DATA_PATH / "hypotheses_by_cluster_summary.csv"
    if hypotheses_path.exists():
        df = pd.read_csv(hypotheses_path)
        # Очищаем данные от NaN и Inf для JSON
        records = []
        for _, row in df.iterrows():
            record = {}
            for col, val in row.items():
                if isinstance(val, float):
                    if pd.isna(val) or np.isinf(val):
                        record[col] = None
                    else:
                        record[col] = val
                else:
                    record[col] = val
            records.append(record)
        return TestsResultsResponse(hypotheses=records)
    
    return TestsResultsResponse(hypotheses=[])


# =====================================================
# ЭНДПОИНТЫ: ОТЧЁТЫ (Шаг 6)
# =====================================================

@app.post(
    "/api/report/generate",
    response_model=ReportGenerateResponse,
    summary="Генерация итогового отчёта"
)
async def generate_report():
    """Генерация полного отчёта"""
    try:
        logger.info("Начало генерации отчёта...")
        reporter = ReportGenerator(DATA_PATH)
        report = reporter.generate_report()
        state.report_generated = True
        state.current_report = report
        logger.info("Отчёт сгенерирован")
        
        return ReportGenerateResponse(
            status="success",
            message="Отчёт сгенерирован",
            report_path=str(DATA_PATH / "final_report.json")
        )
    except Exception as e:
        logger.error(f"Ошибка генерации отчёта: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/report",
    response_model=Dict[str, Any],
    summary="Получение итогового отчёта"
)
async def get_report():
    """Получить сгенерированный отчёт"""
    if not state.report_generated:
        raise HTTPException(status_code=400, detail="Отчёт не сгенерирован")
    
    report_path = DATA_PATH / "final_report.json"
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            return clean_float_values(json.load(f))
    
    raise HTTPException(status_code=404, detail="Отчёт не найден")


@app.get("/api/report/summary", response_model=Dict[str, Any])
async def get_report_summary():
    """Получить краткую версию отчёта"""
    summary_path = DATA_PATH / "report_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        return {"summary": clean_float_values(df.to_dict('records'))}
    raise HTTPException(status_code=404, detail="Краткий отчёт не найден")


# =====================================================
# ЭНДПОИНТЫ: LLM
# =====================================================

@app.post(
    "/api/llm/analyze",
    response_model=LLMResponse,
    summary="Анализ данных с LLM"
)
async def llm_analyze(request: LLMAnalyzeRequest):
    """Анализ данных с помощью LLM"""
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    
    data_summary = get_data_summary()
    response = state.llm_client.analyze_data(data_summary)
    return LLMResponse(response=response)


@app.post(
    "/api/llm/hypotheses",
    response_model=LLMResponse,
    summary="Генерация гипотез"
)
async def llm_hypotheses():
    """Генерация гипотез на основе данных"""
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    
    data_summary = get_data_summary()
    response = state.llm_client.generate_hypotheses(data_summary)
    return LLMResponse(response=response)


@app.post(
    "/api/llm/strategy",
    response_model=LLMResponse,
    summary="Генерация стратегии"
)
async def llm_strategy():
    """Генерация стратегии продвижения"""
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    
    if not state.clustering_completed:
        raise HTTPException(status_code=400, detail="Сначала выполните кластеризацию")
    
    cluster_info = state.df.groupby('cluster').agg({
        'keyword': 'nunique',
        'motiv': 'mean',
        'position': 'mean'
    }).round(2).to_dict()
    
    response = state.llm_client.generate_strategy(cluster_info)
    return LLMResponse(response=response)


@app.post(
    "/api/llm/explain_clusters",
    response_model=LLMResponse,
    summary="Объяснение кластеров"
)
async def llm_explain_clusters():
    """Объяснение кластеров с LLM"""
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    
    if not state.clustering_completed:
        raise HTTPException(status_code=400, detail="Сначала выполните кластеризацию")
    
    cluster_stats = state.df.groupby('cluster').agg({
        'keyword': 'nunique',
        'motiv': ['mean', 'sum'],
        'position': ['mean', 'min', 'max']
    }).round(2)
    cluster_stats.columns = ['keywords', 'motiv_mean', 'motiv_sum', 'position_mean', 'position_min', 'position_max']
    
    response = state.llm_client.explain_clusters(cluster_stats)
    return LLMResponse(response=response)


@app.post(
    "/api/llm/explain_plot",
    response_model=LLMResponse,
    summary="Объяснение графика"
)
async def llm_explain_plot(request: ExplainPlotRequest):
    """Объяснить график с использованием ASO методологии"""
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    
    response = state.llm_client.explain_plot(
        request.plot_name,
        request.description,
        request.context
    )
    return LLMResponse(response=response)


@app.post(
    "/api/llm/recommendations",
    response_model=LLMResponse,
    summary="ASO-рекомендации"
)
async def llm_recommendations():
    """Получить ASO-рекомендации"""
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    
    data_summary = get_data_summary()
    response = state.llm_client.get_recommendations(data_summary)
    return LLMResponse(response=response)


@app.post(
    "/api/llm/model_explanation",
    response_model=LLMResponse,
    summary="Объяснение ML модели"
)
async def llm_model_explanation():
    """Объяснение ML модели"""
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    
    model_results_path = DATA_PATH / "model_results_final.csv"
    model_results = {}
    if model_results_path.exists():
        df = pd.read_csv(model_results_path)
        model_results = clean_float_values(df.to_dict('records'))
    
    importance_path = DATA_PATH / "feature_importance_final.csv"
    importance = []
    if importance_path.exists():
        df = pd.read_csv(importance_path)
        importance = clean_float_values(df.head(10).to_dict('records'))
    
    data = {
        "model_results": model_results,
        "feature_importance": importance,
        "best_model": model_results[0] if model_results else None
    }
    
    response = state.llm_client.get_model_explanation(data)
    return LLMResponse(response=response)


@app.post(
    "/api/llm/chat",
    response_model=LLMResponse,
    summary="Свободный диалог"
)
async def llm_chat(request: LLMChatRequest):
    """Свободный диалог с LLM"""
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    
    context = request.context
    if not context and state.data_loaded:
        data_summary = get_data_summary()
        context = f"Данные ASO: {data_summary['total_records']} записей, {data_summary['unique_keywords']} ключей"
    
    response = state.llm_client.chat(request.message, context)
    return LLMResponse(response=response)


@app.get(
    "/api/llm/status",
    response_model=LLMStatusResponse,
    summary="Статус LLM"
)
async def llm_status():
    """Получить статус LLM"""
    status = state.llm_client.get_status()
    return LLMStatusResponse(
        is_available=status['is_available'],
        model=status['model'],
        url=status['url']
    )


@app.get(
    "/api/llm/methodology",
    response_model=MethodologyResponse,
    summary="Получить ASO методологию"
)
async def get_methodology():
    """Получить описание ASO методологии"""
    return MethodologyResponse(
        methodology=state.llm_client.ASO_METHODOLOGY,
        version="1.0",
        rules_count=12,
        clusters=[
            "top10_high", "top10_medium", "top10_low", "top10_zero",
            "top50_high", "top50_medium", "top50_low", "top50_zero",
            "above50_high", "above50_medium", "above50_low", "above50_zero"
        ],
        scenarios=[
            {"name": "sharp_growth", "label": "Резкий рост", "threshold": "+30%"},
            {"name": "gentle_growth", "label": "Плавный рост", "threshold": "+10-30%"},
            {"name": "stable", "label": "Стабильный", "threshold": "0%"},
            {"name": "gentle_decline", "label": "Плавный спад", "threshold": "-10-30%"},
            {"name": "sharp_decline", "label": "Резкий спад", "threshold": "-30%"}
        ]
    )


# =====================================================
# ОБЩИЕ ЭНДПОИНТЫ
# =====================================================

@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Главная страница"
)
async def root():
    """Главная страница"""
    html_path = STATIC_PATH / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding='utf-8'))
    
    return HTMLResponse("""
    <html>
        <head>
            <title>ASO Analytics Service</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #667eea; }
                .card { background: #f7fafc; padding: 20px; border-radius: 10px; margin: 20px 0; }
                a { color: #667eea; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>📊 ASO Analytics Service</h1>
            <p>Сервис для анализа ASO данных приложения Sound Amplifier</p>
            
            <div class="card">
                <h3>🚀 Быстрый старт</h3>
                <ol>
                    <li>Поместите Excel файлы в папку <code>data/</code></li>
                    <li>Нажмите <strong>"Загрузить данные"</strong></li>
                    <li>Выполните <strong>"Все шаги"</strong> для полного анализа</li>
                </ol>
            </div>
            
            <div class="card">
                <h3>📚 Документация API</h3>
                <p><a href="/docs">Swagger UI</a> — интерактивная документация</p>
                <p><a href="/redoc">ReDoc</a> — альтернативная документация</p>
            </div>
            
            <div class="card">
                <h3>🔗 Полезные ссылки</h3>
                <p><a href="/ml">ML страница</a></p>
                <p><a href="/api/plots/all_grouped">Графики по шагам</a></p>
                <p><a href="/api/llm/methodology">ASO методология</a></p>
            </div>
            
            <p style="color: #a0aec0; margin-top: 40px; font-size: 14px;">
                ASO Analytics Service v3.0 | Sound Amplifier
            </p>
        </body>
    </html>
    """)


@app.get(
    "/ml",
    response_class=HTMLResponse,
    summary="ML страница"
)
async def ml_page():
    """Страница с ML аналитикой"""
    html_path = STATIC_PATH / "ml.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding='utf-8'))
    
    return HTMLResponse("""
    <html>
        <head>
            <title>ML Analytics</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #667eea; }
                .card { background: #f7fafc; padding: 20px; border-radius: 10px; margin: 20px 0; }
                a { color: #667eea; text-decoration: none; }
            </style>
        </head>
        <body>
            <h1>🤖 ML Analytics</h1>
            <p>Результаты машинного обучения</p>
            
            <div class="card">
                <h3>📊 API эндпоинты</h3>
                <ul>
                    <li><code>GET /api/ml/results</code> — результаты моделей</li>
                    <li><code>GET /api/ml/feature_importance</code> — важность признаков</li>
                    <li><code>POST /api/ml/predict</code> — предсказание</li>
                </ul>
            </div>
            
            <div class="card">
                <h3>📈 Графики</h3>
                <p><a href="/plots/model_comparison_with_regularization.png">Сравнение моделей</a></p>
                <p><a href="/plots/feature_importance_by_group.png">Важность признаков</a></p>
                <p><a href="/plots/predictions_vs_actual.png">Факт vs Прогноз</a></p>
            </div>
            
            <p><a href="/">← На главную</a></p>
        </body>
    </html>
    """)


@app.get(
    "/plots/{filename}",
    response_class=FileResponse,
    summary="Получение графика"
)
async def get_plot(filename: str):
    """Получить график по имени"""
    plot_path = PLOTS_PATH / filename
    if plot_path.exists():
        return FileResponse(
            plot_path,
            media_type="image/png",
            filename=filename
        )
    raise HTTPException(status_code=404, detail="График не найден")


@app.get(
    "/api/status",
    response_model=StatusResponse,
    summary="Общий статус системы"
)
async def get_system_status():
    """Получить общий статус системы"""
    return StatusResponse(
        data_loaded=state.data_loaded,
        eda_completed=state.eda_completed,
        clustering_completed=state.clustering_completed,
        ml_completed=state.ml_completed,
        tests_completed=state.tests_completed,
        report_generated=state.report_generated,
        llm_available=state.llm_client.is_available,
        records=len(state.df) if state.df is not None else 0,
        keywords=state.df['keyword'].nunique() if state.df is not None else 0
    )


@app.get("/api/health", response_model=Dict[str, str])
async def health_check():
    """Проверка работоспособности"""
    return {
        "status": "healthy",
        "service": "ASO Analytics Service",
        "version": "3.0"
    }


# =====================================================
# ЭКСПОРТ ДАННЫХ
# =====================================================

@app.post("/api/export", response_model=Dict[str, Any])
async def export_data(request: ExportRequest):
    """Экспорт данных в различных форматах"""
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    
    df = state.df.copy()
    
    if request.filters:
        if request.filters.keyword:
            df = df[df['keyword'].str.contains(request.filters.keyword, case=False)]
        if request.filters.cluster:
            df = df[df['cluster'] == request.filters.cluster]
        if request.filters.date_from:
            df = df[df['date'] >= request.filters.date_from]
        if request.filters.date_to:
            df = df[df['date'] <= request.filters.date_to]
    
    if request.columns:
        df = df[request.columns]
    
    export_path = DATA_PATH / f"export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    if request.format == 'csv':
        path = f"{export_path}.csv"
        df.to_csv(path, index=False)
    elif request.format == 'json':
        path = f"{export_path}.json"
        df.to_json(path, orient='records', force_ascii=False)
    elif request.format == 'excel':
        path = f"{export_path}.xlsx"
        df.to_excel(path, index=False)
    else:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат")
    
    return {
        "status": "success",
        "file": str(path),
        "records": len(df),
        "format": request.format
    }


# =====================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# =====================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚀 ASO Analytics Service v3.0                             ║
║                                                              ║
║   Сервис для анализа ASO данных приложения Sound Amplifier   ║
║                                                              ║
║   API документация: http://localhost:8000/docs              ║
║   Главная страница:  http://localhost:8000                  ║
║   ML страница:        http://localhost:8000/ml              ║
║   Графики по шагам:   http://localhost:8000/api/plots/all_grouped ║
║   ASO методология:    http://localhost:8000/api/llm/methodology ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )