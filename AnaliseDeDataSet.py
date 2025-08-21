import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

class DataAnalyzer:
    def __init__(self):
        self.train_df = None
        self.validation_df = None
        self.test_df = None
        self.combined_df = None
        self.actual_columns = []
        
    def load_and_analyze_structure(self):
        print("📂 Carregando e analisando estrutura dos dados...")
        
        try:
            self.train_df = pd.read_csv('C:projeto_completo/dataset/train.csv')
            self.validation_df = pd.read_csv('C:projeto_completo/dataset/validation.csv')
            self.test_df = pd.read_csv('C:projeto_completo/dataset/test.csv')
            
            self.actual_columns = self.train_df.columns.tolist()
            
            print(f"✅ Dados carregados com sucesso!")
            print(f"📋 Colunas encontradas: {self.actual_columns}")
            print(f"📊 Formatos: Treino {self.train_df.shape}, Validação {self.validation_df.shape}, Teste {self.test_df.shape}")
            
            self.train_df['dataset'] = 'train'
            self.validation_df['dataset'] = 'validation'
            self.test_df['dataset'] = 'test'
            
            self.combined_df = pd.concat([self.train_df, self.validation_df, self.test_df], ignore_index=True)
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def basic_info(self):
        
        print("\n" + "="*60)
        print("📋 INFORMAÇÕES BÁSICAS DO DATASET")
        print("="*60)
        
        print(f"\n🎯 VARIÁVEL TARGET: {'income' if 'income' in self.actual_columns else 'NÃO ENCONTRADA'}")
        
        if 'income' in self.combined_df.columns:
            target_dist = self.combined_df['income'].value_counts()
            print(f"Distribuição do target: {target_dist.to_dict()}")
            print(f"Proporção: {target_dist[0]/target_dist.sum():.2%} vs {target_dist[1]/target_dist.sum():.2%}")
        
        print("\n🔍 PRIMEIRAS 3 LINHAS:")
        print(self.combined_df.head(3))
        
        print("\n📊 TIPOS DE DADOS:")
        print(self.combined_df.dtypes)
        
        print("\n📈 ESTATÍSTICAS NUMÉRICAS:")
        numeric_cols = self.combined_df.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'dataset']
        if len(numeric_cols) > 0:
            print(self.combined_df[numeric_cols].describe())
        else:
            print("❌ Nenhuma coluna numérica encontrada")
    
    def analyze_features(self):
        print("\n" + "="*60)
        print("📊 ANÁLISE DETALHADA DAS FEATURES")
        print("="*60)
        
        numeric_features = self.combined_df.select_dtypes(include=['int64', 'float64']).columns
        numeric_features = [col for col in numeric_features if col not in ['dataset']]
        
        categorical_features = self.combined_df.select_dtypes(include=['object']).columns
        categorical_features = [col for col in categorical_features if col not in ['dataset', 'income']]
        
        print(f"🔢 Features Numéricas ({len(numeric_features)}): {numeric_features}")
        print(f"🔤 Features Categóricas ({len(categorical_features)}): {categorical_features}")
        
        if numeric_features:
            print("\n" + "="*40)
            print("📈 FEATURES NUMÉRICAS")
            print("="*40)
            
            for feature in numeric_features:
                print(f"\n📊 {feature}:")
                stats = self.combined_df[feature].describe()
                print(f"   Média: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                print(f"   Min: {stats['min']}, Max: {stats['max']}")
                print(f"   Missing: {self.combined_df[feature].isnull().sum()}")
                
                if 'income' in self.combined_df.columns:
                    temp_df = self.combined_df.copy()
                    temp_df['income_encoded'] = temp_df['income'].apply(lambda x: 1 if x == '>50K' else 0)
                    correlation = temp_df[feature].corr(temp_df['income_encoded'])
                    print(f"   Correlação com income: {correlation:.4f}")
        
        if categorical_features:
            print("\n" + "="*40)
            print("📊 FEATURES CATEGÓRICAS")
            print("="*40)
            
            for feature in categorical_features[:5]:
                print(f"\n📊 {feature}:")
                value_counts = self.combined_df[feature].value_counts()
                print(f"   Valores únicos: {len(value_counts)}")
                print(f"   Top 5 valores: {value_counts.head(5).to_dict()}")
                print(f"   Missing: {self.combined_df[feature].isnull().sum()}")
    
    def check_missing_values(self):
        print("\n" + "="*60)
        print("🔍 VALORES MISSING POR COLUNA")
        print("="*60)
        
        missing = self.combined_df.isnull().sum()
        missing_percent = (missing / len(self.combined_df)) * 100
        
        missing_df = pd.DataFrame({
            'Valores Missing': missing,
            'Percentual (%)': missing_percent
        })
        
        missing_cols = missing_df[missing_df['Valores Missing'] > 0]
        
        if len(missing_cols) > 0:
            print("⚠️  COLUNAS COM VALORES MISSING:")
            print(missing_cols)
        else:
            print("✅ Nenhum valor missing encontrado!")
        
        print(f"\n📊 Total de valores missing: {missing.sum()}")
    
    def visualize_distributions(self):
        print("\n" + "="*60)
        print("📊 VISUALIZAÇÃO DAS DISTRIBUIÇÕES")
        print("="*60)
        
        numeric_features = self.combined_df.select_dtypes(include=['int64', 'float64']).columns
        numeric_features = [col for col in numeric_features if col not in ['dataset']]
        
        if len(numeric_features) > 0:
            features_to_plot = numeric_features[:4]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, feature in enumerate(features_to_plot):
                if i < len(axes):
                    self.combined_df[feature].hist(bins=30, ax=axes[i])
                    axes[i].set_title(f'Distribuição de {feature}')
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('Frequência')
            
            plt.tight_layout()
            plt.savefig('C:projeto_completo/results/numeric_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()
        

        if 'income' in self.combined_df.columns:
            plt.figure(figsize=(10, 6))
            self.combined_df['income'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
            plt.title('Distribuição da Variável Target (income)')
            plt.xlabel('Categoria de Renda')
            plt.ylabel('Quantidade')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig('C:projeto_completo/results/target_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def analyze_target_relationship(self):
        if 'income' not in self.combined_df.columns:
            print("❌ Variável target 'income' não encontrada para análise")
            return
        
        print("\n" + "="*60)
        print("📈 RELAÇÃO DAS FEATURES COM O TARGET")
        print("="*60)
        

        numeric_features = self.combined_df.select_dtypes(include=['int64', 'float64']).columns
        numeric_features = [col for col in numeric_features if col not in ['dataset']]
        
        if numeric_features:
            print("🔢 RELAÇÃO DE FEATURES NUMÉRICAS COM INCOME:")
            
            for feature in numeric_features[:3]:
                print(f"\n📊 {feature}:")
                
                stats_by_income = self.combined_df.groupby('income')[feature].agg(['mean', 'std', 'median'])
                print(stats_by_income)
                
                group_0 = self.combined_df[self.combined_df['income'] == '<=50K'][feature].dropna()
                group_1 = self.combined_df[self.combined_df['income'] == '>50K'][feature].dropna()
                
                if len(group_0) > 0 and len(group_1) > 0:
                    t_stat, p_value = stats.ttest_ind(group_0, group_1)
                    print(f"   Teste t: p-value = {p_value:.4f} {'✅' if p_value < 0.05 else '❌'}")
    
    def run_adaptive_analysis(self):
        print("🎯 ANÁLISE EXPLORATÓRIA ADAPTATIVA")
        print("="*80)
        
        
        if not self.load_and_analyze_structure():
            return
        
        
        self.basic_info()
        
        
        self.check_missing_values()
        
        
        self.analyze_features()
        
        
        self.visualize_distributions()
        
        
        self.analyze_target_relationship()
        
        print("\n✅ ANÁLISE CONCLUÍDA!")
        
        
        print("\n" + "="*60)
        print("📋 RESUMO DA ANÁLISE")
        print("="*60)
        print(f"• Total de instâncias: {len(self.combined_df):,}")
        print(f"• Total de features: {len(self.actual_columns)}")
        print(f"• Features numéricas: {len(self.combined_df.select_dtypes(include=['int64', 'float64']).columns) - 1}")
        print(f"• Features categóricas: {len(self.combined_df.select_dtypes(include=['object']).columns) - 2}")  # -2 para dataset e income
        
        if 'income' in self.combined_df.columns:
            target_dist = self.combined_df['income'].value_counts()
            print(f"• Distribuição do target: {target_dist.to_dict()}")
            print(f"• Proporção: {target_dist[0]/target_dist.sum():.2%} vs {target_dist[1]/target_dist.sum():.2%}")


if __name__ == "__main__":
    print("🔍 INICIANDO ANÁLISE DO DATASET...")
    analyzer = DataAnalyzer()
    analyzer.run_adaptive_analysis()