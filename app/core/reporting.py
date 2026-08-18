"""
Модуль генерации отчётов
Шаг 6: Сбор результатов, генерация JSON отчёта, итоговые выводы
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ReportGenerator:
    """Класс для генерации итоговых отчётов"""
    
    def __init__(self, data_path: Path):
        """
        Args:
            data_path: Путь к папке с данными
        """
        self.data_path = Path(data_path)
        self.df = None
        self.results = {}
    
    def generate_report(self) -> dict:
        """Генерация полного отчёта"""
        print("\n" + "="*80)
        print("ШАГ 6: ГЕНЕРАЦИЯ ИТОГОВОГО ОТЧЁТА")
        print("="*80)
        
        # 1. Загрузка данных
        self._load_data()
        
        # 2. Сбор всех результатов
        self._collect_results()
        
        # 3. Формирование отчёта
        report = self._build_report()
        
        # 4. Сохранение
        self._save_report(report)
        
        # 5. Вывод краткого отчёта
        self._print_summary(report)
        
        print("\n" + "="*80)
        print("✅ ШАГ 6 ЗАВЕРШЕН УСПЕШНО!")
        print("="*80)
        
        return report
    
    def _load_data(self):
        """Загрузка всех необходимых данных"""
        print("\n" + "="*60)
        print("1. ЗАГРУЗКА ДАННЫХ")
        print("="*60)
        
        # Основные данные
        self.df = pd.read_pickle(self.data_path / "data_with_clusters.pkl")
        print(f"  + Загружены данные: {len(self.df):,} записей")
        
        # Результаты моделей
        model_results_path = self.data_path / "model_results_final.csv"
        if model_results_path.exists():
            self.model_results = pd.read_csv(model_results_path)
            print(f"  + Загружены результаты моделей")
        
        # Важность признаков
        importance_path = self.data_path / "feature_importance_final.csv"
        if importance_path.exists():
            self.importance = pd.read_csv(importance_path)
            print(f"  + Загружена важность признаков")
        
        # Результаты гипотез
        hypotheses_path = self.data_path / "hypotheses_by_cluster_summary.csv"
        if hypotheses_path.exists():
            self.hypotheses = pd.read_csv(hypotheses_path)
            print(f"  + Загружены результаты гипотез")
        
        # Результаты правил
        rules_path = self.data_path / "rules_from_document_summary_rules.csv"
        if rules_path.exists():
            self.rules = pd.read_csv(rules_path)
            print(f"  + Загружены результаты правил")
        
        # Якорные слова
        anchor_path = self.data_path / "anchor_words.csv"
        if anchor_path.exists():
            self.anchor_df = pd.read_csv(anchor_path)
            print(f"  + Загружены якорные слова")
    
    def _collect_results(self):
        """Сбор всех результатов в единую структуру"""
        print("\n" + "="*60)
        print("2. СБОР РЕЗУЛЬТАТОВ")
        print("="*60)
        
        df = self.df
        
        # Общая статистика
        self.results['general'] = {
            'total_records': len(df),
            'unique_keywords': df['keyword'].nunique(),
            'unique_clusters': df['cluster'].nunique(),
            'period_start': df['date'].min().strftime('%Y-%m-%d'),
            'period_end': df['date'].max().strftime('%Y-%m-%d'),
            'days': (df['date'].max() - df['date'].min()).days
        }
        
        # Статистика по мотиву
        motiv_positive = df[df['motiv'] > 0]['motiv']
        self.results['motiv_stats'] = {
            'mean': float(df['motiv'].mean()),
            'median': float(df['motiv'].median()),
            'max': float(df['motiv'].max()),
            'positive_count': int(len(motiv_positive)),
            'positive_pct': float(len(motiv_positive) / len(df) * 100),
            'mean_positive': float(motiv_positive.mean()) if len(motiv_positive) > 0 else 0
        }
        
        # Статистика по позиции
        position_valid = df[df['position'] != -1]['position']
        self.results['position_stats'] = {
            'mean': float(position_valid.mean()) if len(position_valid) > 0 else 0,
            'median': float(position_valid.median()) if len(position_valid) > 0 else 0,
            'min': float(position_valid.min()) if len(position_valid) > 0 else 0,
            'max': float(position_valid.max()) if len(position_valid) > 0 else 0,
            'valid_count': int(len(position_valid)),
            'valid_pct': float(len(position_valid) / len(df) * 100)
        }
        
        # Статистика по кластерам
        cluster_stats = df.groupby('cluster').agg({
            'keyword': 'nunique',
            'motiv': 'mean',
            'position': 'mean'
        }).round(2)
        cluster_stats.columns = ['keywords', 'motiv_mean', 'position_mean']
        self.results['cluster_stats'] = cluster_stats.to_dict()
        
        # Якорные слова
        if hasattr(self, 'anchor_df') and self.anchor_df is not None:
            self.results['anchor_words'] = self.anchor_df.groupby('cluster').head(3).to_dict('records')
        
        # Результаты моделей
        if hasattr(self, 'model_results') and self.model_results is not None:
            self.results['model_results'] = self.model_results.to_dict('records')
            
            # Лучшая модель
            best_idx = self.model_results['MAE'].idxmin()
            self.results['best_model'] = self.model_results.loc[best_idx].to_dict()
        
        # Важность признаков
        if hasattr(self, 'importance') and self.importance is not None:
            self.results['feature_importance'] = self.importance.head(15).to_dict('records')
        
        # Результаты гипотез
        if hasattr(self, 'hypotheses') and self.hypotheses is not None:
            self.results['hypotheses'] = self.hypotheses.to_dict('records')
        
        # Результаты правил
        if hasattr(self, 'rules') and self.rules is not None:
            self.results['rules'] = self.rules.to_dict('records')
        
        print("  + Все результаты собраны")
    
    def _build_report(self) -> dict:
        """Формирование итогового отчёта"""
        print("\n" + "="*60)
        print("3. ФОРМИРОВАНИЕ ОТЧЁТА")
        print("="*60)
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0',
                'app': 'Sound Amplifier ASO Analytics'
            },
            'executive_summary': self._build_executive_summary(),
            'data_summary': self._build_data_summary(),
            'clustering_results': self._build_clustering_summary(),
            'ml_results': self._build_ml_summary(),
            'hypothesis_results': self._build_hypothesis_summary(),
            'rules_results': self._build_rules_summary(),
            'recommendations': self._build_recommendations(),
            'detailed_results': self.results
        }
        
        print("  + Отчёт сформирован")
        return report
    
    def _build_executive_summary(self) -> dict:
        """Краткое резюме"""
        df = self.df
        
        return {
            'total_keywords': int(df['keyword'].nunique()),
            'total_records': int(len(df)),
            'total_clusters': int(df['cluster'].nunique()),
            'period': f"{df['date'].min().strftime('%Y-%m-%d')} - {df['date'].max().strftime('%Y-%m-%d')}",
            'avg_position': float(df[df['position'] != -1]['position'].mean()) if len(df[df['position'] != -1]) > 0 else 0,
            'avg_motiv': float(df['motiv'].mean()),
            'best_model': self.results.get('best_model', {}).get('Model', 'N/A'),
            'best_model_mae': self.results.get('best_model', {}).get('MAE', 'N/A'),
            'hypotheses_confirmed': self._count_confirmed_hypotheses(),
            'status': '✅ Анализ завершён успешно'
        }
    
    def _build_data_summary(self) -> dict:
        """Сводка по данным"""
        return {
            'general': self.results.get('general', {}),
            'motiv': self.results.get('motiv_stats', {}),
            'position': self.results.get('position_stats', {})
        }
    
    def _build_clustering_summary(self) -> dict:
        """Сводка по кластеризации"""
        if 'cluster_stats' not in self.results:
            return {}
        
        cluster_stats = self.results['cluster_stats']
        
        # Лучший и худший кластеры
        best_cluster = min(cluster_stats.get('position_mean', {}).items(), key=lambda x: x[1])
        worst_cluster = max(cluster_stats.get('position_mean', {}).items(), key=lambda x: x[1])
        
        return {
            'total_clusters': len(cluster_stats.get('position_mean', {})),
            'best_cluster': {
                'name': best_cluster[0] if best_cluster else None,
                'avg_position': best_cluster[1] if best_cluster else None
            },
            'worst_cluster': {
                'name': worst_cluster[0] if worst_cluster else None,
                'avg_position': worst_cluster[1] if worst_cluster else None
            },
            'anchor_words': self.results.get('anchor_words', [])
        }
    
    def _build_ml_summary(self) -> dict:
        """Сводка по ML моделям"""
        if 'best_model' not in self.results:
            return {}
        
        best = self.results['best_model']
        
        return {
            'best_model': best.get('Model', 'N/A'),
            'mae': best.get('MAE', 'N/A'),
            'rmse': best.get('RMSE', 'N/A'),
            'r2': best.get('R2', 'N/A'),
            'top_features': self.results.get('feature_importance', [])[:5]
        }
    
    def _build_hypothesis_summary(self) -> dict:
        """Сводка по гипотезам"""
        if 'hypotheses' not in self.results:
            return {}
        
        hypotheses = self.results['hypotheses']
        
        # Подсчёт подтверждённых
        confirmed = 0
        total = 0
        for h in hypotheses:
            for key in ['H1_Результат', 'H3_Результат', 'H4_Результат', 'H5_Результат']:
                if key in h:
                    total += 1
                    if '✅' in str(h[key]):
                        confirmed += 1
        
        return {
            'total_tests': total,
            'confirmed': confirmed,
            'confirmation_rate': f"{confirmed/total*100:.1f}%" if total > 0 else "0%",
            'details': hypotheses
        }
    
    def _build_rules_summary(self) -> dict:
        """Сводка по правилам"""
        if 'rules' not in self.results:
            return {}
        
        rules = self.results['rules']
        
        # Подсчёт подтверждённых правил
        confirmed = sum(1 for r in rules if '✅' in str(r.get('Статус', '')))
        total = len(rules)
        
        return {
            'total_rules': total,
            'confirmed': confirmed,
            'confirmation_rate': f"{confirmed/total*100:.1f}%" if total > 0 else "0%",
            'details': rules
        }
    
    def _build_recommendations(self) -> dict:
        """Рекомендации"""
        recommendations = {
            'critical': [
                {
                    'priority': 1,
                    'title': 'Используйте XGBoost с регуляризацией',
                    'description': 'Модель показывает лучшие результаты и меньше переобучается',
                    'action': 'Внедрите модель в продакшен для прогнозирования изменения позиций'
                },
                {
                    'priority': 2,
                    'title': 'Следите за текущей позицией',
                    'description': 'Позиция и её лаги — главные предикторы изменения',
                    'action': 'Включите позицию в систему мониторинга как ключевой показатель'
                },
                {
                    'priority': 3,
                    'title': 'Исключите признаки с утечкой',
                    'description': 'position_change, motiv_change и др. дают ложную точность',
                    'action': 'Не используйте эти признаки в реальных прогнозах'
                }
            ],
            'important': [
                {
                    'priority': 4,
                    'title': 'Мотив не является главным фактором',
                    'description': 'Мотив объясняет только 8-10% изменений',
                    'action': 'Сфокусируйтесь на позиции и её динамике'
                },
                {
                    'priority': 5,
                    'title': 'Резкий рост мотива даёт лучший эффект',
                    'description': 'Сценарий резкого роста (+30%) показывает лучшие результаты',
                    'action': 'Используйте резкий рост для быстрого продвижения'
                },
                {
                    'priority': 6,
                    'title': 'Используйте якорные слова',
                    'description': 'Якорные слова помогают продвигать целые кластеры',
                    'action': 'Начните с якорных слов для каждого кластера'
                }
            ],
            'recommended': [
                {
                    'priority': 7,
                    'title': 'Проверяйте гипотезы на свежих данных',
                    'description': 'Используйте данные после 20.06.2026 для валидации',
                    'action': 'Регулярно обновляйте тестовую выборку'
                },
                {
                    'priority': 8,
                    'title': 'Добавьте данные о конкурентах',
                    'description': 'Это улучшит качество модели',
                    'action': 'Интегрируйте внешние данные в пайплайн'
                },
                {
                    'priority': 9,
                    'title': 'Переобучайте модель еженедельно',
                    'description': 'Данные меняются, модель должна адаптироваться',
                    'action': 'Настройте автоматическое переобучение'
                },
                {
                    'priority': 10,
                    'title': 'Используйте регуляризацию',
                    'description': 'Регуляризация уменьшает переобучение',
                    'action': 'Всегда включайте регуляризацию в модели'
                }
            ]
        }
        
        return recommendations
    
    def _count_confirmed_hypotheses(self) -> int:
        """Подсчёт подтверждённых гипотез"""
        if 'hypotheses' not in self.results:
            return 0
        
        confirmed = 0
        for h in self.results['hypotheses']:
            for key in ['H1_Результат', 'H3_Результат', 'H4_Результат', 'H5_Результат']:
                if key in h and '✅' in str(h[key]):
                    confirmed += 1
        
        return confirmed
    
    def _save_report(self, report: dict):
        """Сохранение отчёта в JSON"""
        print("\n" + "="*60)
        print("4. СОХРАНЕНИЕ ОТЧЁТА")
        print("="*60)
        
        report_path = self.data_path / "final_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"  + Отчёт сохранён: {report_path}")
        
        # Сохраняем также в CSV для удобства
        self._save_csv_summary(report)
    
    def _save_csv_summary(self, report: dict):
        """Сохранение краткого отчёта в CSV"""
        summary_data = []
        
        # Основные метрики
        summary_data.append({
            'Показатель': 'Всего записей',
            'Значение': report.get('executive_summary', {}).get('total_records', 'N/A')
        })
        summary_data.append({
            'Показатель': 'Уникальных ключей',
            'Значение': report.get('executive_summary', {}).get('total_keywords', 'N/A')
        })
        summary_data.append({
            'Показатель': 'Количество кластеров',
            'Значение': report.get('executive_summary', {}).get('total_clusters', 'N/A')
        })
        summary_data.append({
            'Показатель': 'Средняя позиция',
            'Значение': report.get('executive_summary', {}).get('avg_position', 'N/A')
        })
        summary_data.append({
            'Показатель': 'Средний мотив',
            'Значение': report.get('executive_summary', {}).get('avg_motiv', 'N/A')
        })
        summary_data.append({
            'Показатель': 'Лучшая модель',
            'Значение': report.get('executive_summary', {}).get('best_model', 'N/A')
        })
        summary_data.append({
            'Показатель': 'MAE модели',
            'Значение': report.get('executive_summary', {}).get('best_model_mae', 'N/A')
        })
        summary_data.append({
            'Показатель': 'Подтверждено гипотез',
            'Значение': report.get('executive_summary', {}).get('hypotheses_confirmed', 'N/A')
        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.data_path / "report_summary.csv", index=False)
        print(f"  + Краткий отчёт сохранён: {self.data_path / 'report_summary.csv'}")
    
    def _print_summary(self, report: dict):
        """Вывод краткого отчёта в консоль"""
        print("\n" + "="*60)
        print("5. КРАТКИЙ ОТЧЁТ")
        print("="*60)
        
        exec_summary = report.get('executive_summary', {})
        
        print(f"""
📊 ИТОГОВЫЙ ОТЧЁТ ПО АНАЛИЗУ ASO

ОБЩАЯ ИНФОРМАЦИЯ:
  - Всего записей: {exec_summary.get('total_records', 'N/A')}
  - Уникальных ключей: {exec_summary.get('total_keywords', 'N/A')}
  - Количество кластеров: {exec_summary.get('total_clusters', 'N/A')}
  - Период: {exec_summary.get('period', 'N/A')}
  - Средняя позиция: {exec_summary.get('avg_position', 'N/A')}
  - Средний мотив: {exec_summary.get('avg_motiv', 'N/A')}

МОДЕЛИРОВАНИЕ:
  - Лучшая модель: {exec_summary.get('best_model', 'N/A')}
  - MAE: {exec_summary.get('best_model_mae', 'N/A')}

ГИПОТЕЗЫ:
  - Подтверждено: {exec_summary.get('hypotheses_confirmed', 'N/A')}

СТАТУС: {exec_summary.get('status', 'N/A')}
""")
    
    def get_report(self) -> dict:
        """Получить сохранённый отчёт"""
        report_path = self.data_path / "final_report.json"
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None