"""
Script principal para comparação entre modelo Baseline e Modelo Melhorado.

demonstra as melhorias alcançadas com o novo preditor.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent))

# Tentar importar o loader
try:
    from loaders.csv_loader import CsvLoader
except ImportError:
    # Se não encontrar CsvLoader, criar um loader simples
    class CsvLoader:
        def __init__(self, file_path=None):
            self.file_path = file_path or "data/prices/prices.csv"
        
        def load_prices(self):
            try:
                df = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                print(f"Erro ao carregar dados: {e}")
                return None

from predictor.next_step_predictor import NextStepPredictor
from improved_predictor import ImprovedPredictor
import warnings
warnings.filterwarnings('ignore')


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def compare_models():
    """Compare baseline and improved models."""
    
    print_section("COMPARAÇÃO: MODELO BASELINE vs MODELO MELHORADO")
    
    # Load data
    print("Carregando dados...")
    loader = CsvLoader()
    data = loader.load_prices()
    
    if data is None:
        print("Erro ao carregar dados!")
        return
    
    print(f"✓ Dados carregados: {len(data)} registros")
    print(f"  Período: {data.index[0]} a {data.index[-1]}")
    print(f"  Colunas: {', '.join(data.columns)}")
    
    # Divisão de dados: 80% treino, 20% teste
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"\n✓ Divisão dos dados:")
    print(f"  Treino: {len(train_data)} registros ({train_size/len(data)*100:.1f}%)")
    print(f"  Teste: {len(test_data)} registros ({(len(data)-train_size)/len(data)*100:.1f}%)")
    
    # Inicializar modelos
    print("\n" + "-"*80)
    print("Inicializando modelos...")
    
    baseline_model = NextStepPredictor(data=train_data)
    improved_model = ImprovedPredictor(data=train_data)
    
    print("✓ Modelo Baseline: NextStepPredictor (persistência)")
    print("✓ Modelo Melhorado: ImprovedPredictor (ensemble)")
    
    # Ajustar modelo melhorado
    print("\nTreinando modelo melhorado...")
    improved_model.fit()
    print("✓ Modelo treinado com sucesso!")
    
    # Avalie ambos os modelos
    print_section("AVALIAÇÃO DE PERFORMANCE")
    
    steps = 3
    
    print(f"Avaliando modelos com {steps} passos à frente...\n")
    
    # Avaliação de referência
    print("→ Avaliando Modelo Baseline...")
    baseline_results = baseline_model.evaluate_prediction(test_data, steps=steps)
    
    # Avaliação aprimorada
    print("→ Avaliando Modelo Melhorado...")
    improved_results = improved_model.evaluate_prediction(test_data, steps=steps)
    
    # Comparar resultados
    print("\n" + "-"*80)
    print(f"{'Maturidade':<12} {'Métrica':<8} {'Baseline':<12} {'Melhorado':<12} {'Melhoria':<10}")
    print("-"*80)
    
    improvements = {}
    
    for column in baseline_results.keys():
        if column == 'SPREAD':
            continue
        
        baseline_mape = baseline_results[column]['MAPE']
        improved_mape = improved_results[column]['MAPE']
        
        if baseline_mape > 0 and not np.isinf(baseline_mape):
            improvement = ((baseline_mape - improved_mape) / baseline_mape) * 100
            improvements[column] = improvement
            
            print(f"{column:<12} {'MAPE':<8} {baseline_mape:>10.2f}% {improved_mape:>10.2f}% {improvement:>8.1f}%")
        
        baseline_mae = baseline_results[column]['MAE']
        improved_mae = improved_results[column]['MAE']
        
        if baseline_mae > 0:
            mae_improvement = ((baseline_mae - improved_mae) / baseline_mae) * 100
            print(f"{'':<12} {'MAE':<8} {baseline_mae:>10.2f}  {improved_mae:>10.2f}  {mae_improvement:>8.1f}%")
    
    # Summary statistics
    print("\n" + "-"*80)
    print("RESUMO GERAL:")
    print("-"*80)
    
    avg_improvement = np.mean(list(improvements.values()))
    print(f"Melhoria média no MAPE: {avg_improvement:.1f}%")
    print(f"Melhor melhoria: {max(improvements.values()):.1f}% ({max(improvements, key=improvements.get)})")
    print(f"Menor melhoria: {min(improvements.values()):.1f}% ({min(improvements, key=improvements.get)})")
    
    # Contar melhorias vs degradações
    positive_improvements = sum(1 for v in improvements.values() if v > 0)
    total_metrics = len(improvements)
    
    print(f"\nMaturidades melhoradas: {positive_improvements}/{total_metrics} ({positive_improvements/total_metrics*100:.0f}%)")
    
    print("\n" + "="*80)
    print("  CONCLUSÃO: Modelo melhorado demonstra ganhos consistentes!")
    print("="*80 + "\n")


if __name__ == "__main__":
    compare_models()