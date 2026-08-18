"""
Модуль статистических тестов
Шаг 5: Проверка гипотез с разбивкой по кластерам
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, kruskal, ttest_rel
import warnings
warnings.filterwarnings('ignore')


class StatisticalTester:
    """Класс для статистических тестов по кластерам"""
    
    def __init__(self, data_path: Path, df: pd.DataFrame = None):
        """
        Args:
            data_path: Путь к папке с данными
            df: DataFrame с данными (если None, загружается из data_with_clusters.pkl)
        """
        self.data_path = Path(data_path)
        self.plots_path = self.data_path / "plots"
        self.plots_path.mkdir(exist_ok=True)
        
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_pickle(self.data_path / "data_with_clusters.pkl")
        
        self.start_date = pd.to_datetime('2026-06-20')
        self.df_test = None
        self.clusters = []
        self.all_hypotheses = {'H1': {}, 'H3': {}, 'H4': {}, 'H5': {}}
        
        self.scenario_names = {
            'stable': 'стабильный',
            'sharp_growth': 'резкий рост',
            'sharp_decline': 'резкий спад',
            'gentle_growth': 'плавный рост',
            'gentle_decline': 'плавный спад'
        }
    
    def run_tests(self) -> dict:
        """Запуск всех статистических тестов"""
        print("\n" + "="*80)
        print("ШАГ 5: СТАТИСТИЧЕСКИЕ ТЕСТЫ (С РАЗБИВКОЙ ПО КЛАСТЕРАМ)")
        print("="*80)
        
        df = self.df
        
        print(f"\nРазмер данных: {len(df):,} записей")
        print(f"Уникальных ключей: {df['keyword'].nunique()}")
        print(f"Количество кластеров: {df['cluster'].nunique()}")
        
        print("\n✅ Все статистические библиотеки импортированы")
        
        # 1. Фильтрация данных
        self._filter_data()
        
        # 2. Проверка гипотез
        self._test_all_hypotheses()
        
        # 3. Сводная таблица
        self._create_summary_table()
        
        # 4. Визуализация
        self._plot_results()
        
        # 5. Сохранение
        self._save_results()
        
        print("\n" + "="*80)
        print("✅ ШАГ 5 ЗАВЕРШЕН УСПЕШНО!")
        print("="*80)
        
        return self.all_hypotheses
    
    def _filter_data(self):
        """Фильтрация данных после 20.06.2026"""
        print("\n" + "="*60)
        print("ФИЛЬТРАЦИЯ ДАННЫХ")
        print("="*60)
        
        self.df_test = self.df[self.df['date'] >= self.start_date].copy()
        
        print(f"\n  Исходный размер данных: {len(self.df):,} записей")
        print(f"  Данных с 20.06.2026: {len(self.df_test):,} записей")
        print(f"  Период: {self.df_test['date'].min()} - {self.df_test['date'].max()}")
        print(f"  Уникальных ключей: {self.df_test['keyword'].nunique()}")
        
        if len(self.df_test) == 0:
            print("\n  ⚠️ Нет данных после 20.06.2026! Используем последние 30% данных")
            self.df_test = self.df.sort_values('date').tail(int(len(self.df) * 0.3))
            print(f"  Взято последних 30% данных: {len(self.df_test):,} записей")
        
        self.clusters = [c for c in self.df_test['cluster'].unique() if c != 'nan' and not pd.isna(c)]
        print(f"\n  Активных кластеров в тестовом периоде: {len(self.clusters)}")
        print(f"  Кластеры: {sorted(self.clusters)}")
    
    def _test_all_hypotheses(self):
        """Проверка всех гипотез по кластерам"""
        print("\n" + "="*60)
        print("ПРОВЕРКА ГИПОТЕЗ ПО КЛАСТЕРАМ")
        print("="*60)
        
        for cluster in sorted(self.clusters):
            print(f"\n{'='*50}")
            print(f"КЛАСТЕР: {cluster}")
            print(f"{'='*50}")
            
            df_cluster = self.df_test[self.df_test['cluster'] == cluster]
            
            # H1: Похожие запросы
            result_h1 = self._test_hypothesis_1(df_cluster, cluster)
            if result_h1:
                self.all_hypotheses['H1'][cluster] = result_h1
                print(f"\n  📊 ГИПОТЕЗА 1 (Похожие запросы):")
                print(f"    - Ключей в кластере: {result_h1['n_keywords']}")
                print(f"    - Пар для корреляции: {result_h1['n_pairs']}")
                print(f"    - Средняя корреляция: {result_h1['mean_corr']:.3f}")
                print(f"    - p-value: {result_h1['p_value']:.4f}")
                print(f"    - Результат: {result_h1['result']}")
            else:
                print(f"\n  📊 ГИПОТЕЗА 1: Недостаточно данных")
            
            # H3: Эффективность сценариев
            result_h3 = self._test_hypothesis_3(df_cluster, cluster)
            if result_h3:
                self.all_hypotheses['H3'][cluster] = result_h3
                print(f"\n  📊 ГИПОТЕЗА 3 (Эффективность сценариев):")
                print(f"    - Записей: {result_h3['n_records']}")
                print(f"    - Сценариев: {result_h3['n_scenarios']}")
                print(f"    - H-statistic: {result_h3['h_stat']:.4f}")
                print(f"    - p-value: {result_h3['p_value']:.4f}")
                print(f"    - Лучший сценарий: {self.scenario_names.get(result_h3['best_scenario'], result_h3['best_scenario'])}")
                print(f"    - Результат: {result_h3['result']}")
            else:
                print(f"\n  📊 ГИПОТЕЗА 3: Недостаточно данных")
            
            # H4: Влияние фризов
            result_h4 = self._test_hypothesis_4(df_cluster, cluster)
            if result_h4:
                self.all_hypotheses['H4'][cluster] = result_h4
                print(f"\n  📊 ГИПОТЕЗА 4 (Влияние фризов):")
                print(f"    - Записей с нулевым мотивом: {result_h4['n_records']}")
                print(f"    - Ключей анализируемых: {result_h4['n_keywords']}")
                print(f"    - Среднее изменение: {result_h4['mean_diff']:.2f}")
                print(f"    - t-statistic: {result_h4['t_stat']:.4f}")
                print(f"    - p-value: {result_h4['p_value']:.4f}")
                print(f"    - Результат: {result_h4['result']}")
            else:
                print(f"\n  📊 ГИПОТЕЗА 4: Недостаточно данных")
            
            # H5: Влияние мотива
            result_h5 = self._test_hypothesis_5(df_cluster, cluster)
            if result_h5:
                self.all_hypotheses['H5'][cluster] = result_h5
                print(f"\n  📊 ГИПОТЕЗА 5 (Влияние мотива):")
                print(f"    - Записей: {result_h5['n_records']}")
                print(f"    - Pearson корреляция: {result_h5['pearson_corr']:.4f} (p={result_h5['p_value_pearson']:.4f})")
                print(f"    - Spearman корреляция: {result_h5['spearman_corr']:.4f} (p={result_h5['p_value_spearman']:.4f})")
                print(f"    - Cohen's d: {result_h5['cohens_d']:.3f}")
                print(f"    - Результат: {result_h5['result']}")
            else:
                print(f"\n  📊 ГИПОТЕЗА 5: Недостаточно данных")
    
    def _test_hypothesis_1(self, df_cluster, cluster):
        """Гипотеза 1: Корреляция позиций похожих ключей > 0.7"""
        cluster_keywords = df_cluster['keyword'].unique()
        if len(cluster_keywords) < 3:
            return None
        
        top5 = df_cluster['keyword'].value_counts().head(5).index.tolist()
        if len(top5) < 2:
            return None
        
        df_cluster_top = df_cluster[df_cluster['keyword'].isin(top5)]
        
        pivot_position = df_cluster_top.pivot_table(
            index='date', 
            columns='keyword', 
            values='position'
        ).dropna(axis=1, how='all')
        
        if pivot_position.shape[1] < 2:
            return None
        
        pivot_position = pivot_position.fillna(pivot_position.mean())
        corr_matrix = pivot_position.corr()
        
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_values = upper_tri.stack().values
        corr_values = corr_values[~np.isnan(corr_values)]
        
        if len(corr_values) == 0:
            return None
        
        t_stat, p_value = stats.ttest_ind(corr_values, [0.7] * len(corr_values), alternative='less')
        
        return {
            'cluster': cluster,
            'n_keywords': len(cluster_keywords),
            'n_pairs': len(corr_values),
            'mean_corr': np.mean(corr_values),
            'median_corr': np.median(corr_values),
            'min_corr': np.min(corr_values),
            'max_corr': np.max(corr_values),
            't_stat': t_stat,
            'p_value': p_value,
            'hypothesis_accepted': p_value < 0.05,
            'result': '✅ ПОДТВЕРЖДЕНА' if p_value < 0.05 else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _test_hypothesis_3(self, df_cluster, cluster):
        """Гипотеза 3: Различия между сценариями продвижения"""
        df_scenarios = df_cluster[df_cluster['motiv_scenario'] != 'stable'].copy()
        df_scenarios = df_scenarios.dropna(subset=['position_delta_7d'])
        
        if len(df_scenarios) < 10:
            return None
        
        scenario_stats = df_scenarios.groupby('motiv_scenario').agg({
            'position_delta_7d': ['mean', 'std', 'count']
        }).round(3)
        scenario_stats.columns = ['mean_delta', 'std_delta', 'count']
        
        groups = [group['position_delta_7d'].dropna().values for name, group in df_scenarios.groupby('motiv_scenario')]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return None
        
        h_stat, p_value = kruskal(*groups)
        
        if len(scenario_stats) > 0:
            best_scenario = scenario_stats['mean_delta'].idxmin()
        else:
            best_scenario = None
        
        return {
            'cluster': cluster,
            'n_records': len(df_scenarios),
            'n_scenarios': len(scenario_stats),
            'scenario_stats': scenario_stats,
            'h_stat': h_stat,
            'p_value': p_value,
            'best_scenario': best_scenario,
            'hypothesis_accepted': p_value < 0.05,
            'result': '✅ ПОДТВЕРЖДЕНА' if p_value < 0.05 else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _test_hypothesis_4(self, df_cluster, cluster):
        """Гипотеза 4: Изменение позиции после фриза ≠ 0"""
        df_zero_motiv = df_cluster[df_cluster['motiv'] == 0].copy()
        df_zero_motiv = df_zero_motiv.sort_values(['keyword', 'date'])
        
        if len(df_zero_motiv) < 20:
            return None
        
        results = []
        keywords = df_zero_motiv['keyword'].unique()[:5]
        
        for keyword in keywords:
            keyword_data = df_zero_motiv[df_zero_motiv['keyword'] == keyword].sort_values('date')
            if len(keyword_data) > 10:
                first_positions = keyword_data.head(5)['position'].mean()
                last_positions = keyword_data.tail(5)['position'].mean()
                diff = last_positions - first_positions
                results.append({
                    'keyword': keyword,
                    'first_positions': first_positions,
                    'last_positions': last_positions,
                    'diff': diff
                })
        
        if len(results) < 2:
            return None
        
        diff_df = pd.DataFrame(results)
        t_stat, p_value = ttest_rel(diff_df['first_positions'], diff_df['last_positions'])
        
        return {
            'cluster': cluster,
            'n_records': len(df_zero_motiv),
            'n_keywords': len(results),
            'mean_diff': diff_df['diff'].mean(),
            'median_diff': diff_df['diff'].median(),
            'std_diff': diff_df['diff'].std(),
            't_stat': t_stat,
            'p_value': p_value,
            'hypothesis_accepted': p_value < 0.05,
            'result': '✅ ПОДТВЕРЖДЕНА' if p_value < 0.05 else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _test_hypothesis_5(self, df_cluster, cluster):
        """Гипотеза 5: Мотив влияет на изменение позиции"""
        df_analysis = df_cluster.dropna(subset=['position_delta_7d'])
        
        if len(df_analysis) < 20:
            return None
        
        pearson_corr, p_value_pearson = pearsonr(df_analysis['motiv'], df_analysis['position_delta_7d'])
        spearman_corr, p_value_spearman = spearmanr(df_analysis['motiv'], df_analysis['position_delta_7d'])
        
        motiv_positive = df_analysis[df_analysis['motiv'] > 0]['position_delta_7d']
        motiv_zero = df_analysis[df_analysis['motiv'] == 0]['position_delta_7d']
        
        if len(motiv_positive) > 0 and len(motiv_zero) > 0:
            stat, p_value_mw = mannwhitneyu(motiv_positive, motiv_zero)
            n1, n2 = len(motiv_positive), len(motiv_zero)
            mean1, mean2 = motiv_positive.mean(), motiv_zero.mean()
            std1, std2 = motiv_positive.std(), motiv_zero.std()
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        else:
            p_value_mw = 1.0
            cohens_d = 0
        
        return {
            'cluster': cluster,
            'n_records': len(df_analysis),
            'pearson_corr': pearson_corr,
            'p_value_pearson': p_value_pearson,
            'spearman_corr': spearman_corr,
            'p_value_spearman': p_value_spearman,
            'p_value_mw': p_value_mw,
            'cohens_d': cohens_d,
            'hypothesis_accepted': p_value_pearson < 0.05 or p_value_spearman < 0.05,
            'result': '✅ ПОДТВЕРЖДЕНА' if (p_value_pearson < 0.05 or p_value_spearman < 0.05) else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _create_summary_table(self):
        """Создание сводной таблицы"""
        print("\n" + "="*60)
        print("СВОДНАЯ ТАБЛИЦА ПО КЛАСТЕРАМ")
        print("="*60)
        
        summary_data = []
        for cluster in sorted(self.clusters):
            row = {
                'Кластер': cluster,
                'Ключей в кластере': len(self.df_test[self.df_test['cluster'] == cluster]['keyword'].unique())
            }
            
            # H1
            if cluster in self.all_hypotheses['H1']:
                row['H1_Средняя_корреляция'] = self.all_hypotheses['H1'][cluster]['mean_corr']
                row['H1_p-value'] = self.all_hypotheses['H1'][cluster]['p_value']
                row['H1_Результат'] = self.all_hypotheses['H1'][cluster]['result']
            else:
                row['H1_Средняя_корреляция'] = np.nan
                row['H1_p-value'] = np.nan
                row['H1_Результат'] = '⚠️ НЕТ ДАННЫХ'
            
            # H3
            if cluster in self.all_hypotheses['H3']:
                row['H3_Лучший_сценарий'] = self.scenario_names.get(
                    self.all_hypotheses['H3'][cluster]['best_scenario'], 
                    self.all_hypotheses['H3'][cluster]['best_scenario']
                )
                row['H3_p-value'] = self.all_hypotheses['H3'][cluster]['p_value']
                row['H3_Результат'] = self.all_hypotheses['H3'][cluster]['result']
            else:
                row['H3_Лучший_сценарий'] = 'N/A'
                row['H3_p-value'] = np.nan
                row['H3_Результат'] = '⚠️ НЕТ ДАННЫХ'
            
            # H4
            if cluster in self.all_hypotheses['H4']:
                row['H4_Среднее_изменение'] = self.all_hypotheses['H4'][cluster]['mean_diff']
                row['H4_p-value'] = self.all_hypotheses['H4'][cluster]['p_value']
                row['H4_Результат'] = self.all_hypotheses['H4'][cluster]['result']
            else:
                row['H4_Среднее_изменение'] = np.nan
                row['H4_p-value'] = np.nan
                row['H4_Результат'] = '⚠️ НЕТ ДАННЫХ'
            
            # H5
            if cluster in self.all_hypotheses['H5']:
                row['H5_Pearson'] = self.all_hypotheses['H5'][cluster]['pearson_corr']
                row['H5_p-value'] = self.all_hypotheses['H5'][cluster]['p_value_pearson']
                row['H5_Результат'] = self.all_hypotheses['H5'][cluster]['result']
            else:
                row['H5_Pearson'] = np.nan
                row['H5_p-value'] = np.nan
                row['H5_Результат'] = '⚠️ НЕТ ДАННЫХ'
            
            summary_data.append(row)
        
        self.summary_df = pd.DataFrame(summary_data)
        
        # Сохраняем
        self.summary_df.to_csv(self.data_path / "hypotheses_by_cluster_summary.csv", index=False)
        print(f"\n  + Сводная таблица сохранена: {self.data_path / 'hypotheses_by_cluster_summary.csv'}")
    
    def _plot_results(self):
        """Визуализация результатов тестов"""
        print("\n" + "="*60)
        print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ПО КЛАСТЕРАМ")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Проверка гипотез по кластерам', fontsize=16, fontweight='bold')
        
        clusters = sorted([c for c in self.clusters if c in self.all_hypotheses['H1']])
        
        # 1. H1: Средняя корреляция
        ax = axes[0, 0]
        if clusters:
            h1_corrs = [self.all_hypotheses['H1'][c]['mean_corr'] for c in clusters]
            h1_pvalues = [self.all_hypotheses['H1'][c]['p_value'] for c in clusters]
            
            colors = ['green' if p < 0.05 else 'red' for p in h1_pvalues]
            bars = ax.bar(clusters, h1_corrs, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Порог 0.7')
            ax.set_xlabel('Кластер')
            ax.set_ylabel('Средняя корреляция')
            ax.set_title('H1: Похожие запросы')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            for i, (bar, corr, p) in enumerate(zip(bars, h1_corrs, h1_pvalues)):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{corr:.2f}\np={p:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('H1: Похожие запросы')
        
        # 2. H3: p-value
        ax = axes[0, 1]
        clusters_h3 = sorted([c for c in self.clusters if c in self.all_hypotheses['H3']])
        if clusters_h3:
            h3_pvalues = [self.all_hypotheses['H3'][c]['p_value'] for c in clusters_h3]
            colors = ['green' if p < 0.05 else 'red' for p in h3_pvalues]
            bars = ax.bar(clusters_h3, h3_pvalues, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Порог 0.05')
            ax.set_xlabel('Кластер')
            ax.set_ylabel('p-value')
            ax.set_title('H3: Эффективность сценариев')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            for i, (bar, p) in enumerate(zip(bars, h3_pvalues)):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                        f'{p:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('H3: Эффективность сценариев')
        
        # 3. H4: Изменение позиции при фризах
        ax = axes[1, 0]
        clusters_h4 = sorted([c for c in self.clusters if c in self.all_hypotheses['H4']])
        if clusters_h4:
            h4_diffs = [self.all_hypotheses['H4'][c]['mean_diff'] for c in clusters_h4]
            h4_pvalues = [self.all_hypotheses['H4'][c]['p_value'] for c in clusters_h4]
            
            colors = ['green' if p < 0.05 else 'red' for p in h4_pvalues]
            bars = ax.bar(clusters_h4, h4_diffs, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Нет изменения')
            ax.set_xlabel('Кластер')
            ax.set_ylabel('Среднее изменение позиции')
            ax.set_title('H4: Влияние фризов')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            for i, (bar, diff, p) in enumerate(zip(bars, h4_diffs, h4_pvalues)):
                y_offset = 0.5 if diff >= 0 else -0.5
                va = 'bottom' if diff >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + y_offset,
                        f'{diff:.1f}\np={p:.3f}', ha='center', va=va, fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('H4: Влияние фризов')
        
        # 4. H5: Корреляция мотива
        ax = axes[1, 1]
        clusters_h5 = sorted([c for c in self.clusters if c in self.all_hypotheses['H5']])
        if clusters_h5:
            h5_pearson = [self.all_hypotheses['H5'][c]['pearson_corr'] for c in clusters_h5]
            h5_pvalues = [self.all_hypotheses['H5'][c]['p_value_pearson'] for c in clusters_h5]
            
            colors = ['green' if p < 0.05 else 'red' for p in h5_pvalues]
            bars = ax.bar(clusters_h5, h5_pearson, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Кластер')
            ax.set_ylabel('Pearson корреляция')
            ax.set_title('H5: Влияние мотива')
            ax.grid(True, alpha=0.3)
            
            for i, (bar, corr, p) in enumerate(zip(bars, h5_pearson, h5_pvalues)):
                y_offset = 0.02 if corr >= 0 else -0.02
                va = 'bottom' if corr >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + y_offset,
                        f'{corr:.2f}\np={p:.3f}', ha='center', va=va, fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('H5: Влияние мотива')
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'hypotheses_by_cluster.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  + График проверки гипотез по кластерам сохранён: {self.plots_path / 'hypotheses_by_cluster.png'}")
    
    def _save_results(self):
        """Сохранение результатов"""
        print("\n" + "="*60)
        print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*60)
        
        # Сводная таблица уже сохранена в _create_summary_table()
        
        # Краткая таблица результатов гипотез
        summary_simple = self.summary_df[['Кластер', 'Ключей в кластере', 
                                          'H1_Результат', 'H3_Результат', 
                                          'H4_Результат', 'H5_Результат']].copy()
        summary_simple.to_csv(self.data_path / "hypotheses_summary_simple.csv", index=False)
        print(f"  + Краткая таблица сохранена: {self.data_path / 'hypotheses_summary_simple.csv'}")
    
    def get_hypothesis_results(self) -> dict:
        """Получить результаты всех гипотез"""
        return self.all_hypotheses
    
    def get_summary(self) -> pd.DataFrame:
        """Получить сводную таблицу"""
        return self.summary_df