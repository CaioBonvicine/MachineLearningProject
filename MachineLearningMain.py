import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class AdultIncomeML:
    def __init__(self):
        self.preprocessor = None
        self.selector = None
        self.model = None
        self.performance_table = pd.DataFrame(columns=['Etapa', 'Modelo', 'F1_Treino', 'F1_Validacao', 'F1_Teste', 'Melhoria'])
    
    def load_data(self):
        """Carregar dados para treinamento"""
        print("📂 Carregando dados para treinamento...")
        self.train_df = pd.read_csv('C:projeto_completo/dataset/train.csv')
        self.validation_df = pd.read_csv('C:projeto_completo/dataset/validation.csv')
        self.test_df = pd.read_csv('C:projeto_completo/dataset/test.csv')
        print("✅ Dados carregados!")
    def preprocess_data(self):
        """Pré-processar dados para ML"""
        print("🔧 Pré-processando dados...")
        
        # Separar features e target
        X_train = self.train_df.drop('income', axis=1)
        y_train = self.train_df['income'].apply(lambda x: 1 if x == '>50K' else 0)
        
        X_val = self.validation_df.drop('income', axis=1)
        y_val = self.validation_df['income'].apply(lambda x: 1 if x == '>50K' else 0)
        
        X_test = self.test_df.drop('income', axis=1)
        y_test = self.test_df['income'].apply(lambda x: 1 if x == '>50K' else 0)
        
        # Identificar tipos de features
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns
        
        # Criar pré-processador
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Aplicar pré-processamento
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Seleção de features
        self.selector = SelectKBest(f_classif, k=30)
        X_train_selected = self.selector.fit_transform(X_train_processed, y_train)
        X_val_selected = self.selector.transform(X_val_processed)
        X_test_selected = self.selector.transform(X_test_processed)
        
        print(f"📊 Dados processados: {X_train_selected.shape}")
        
        return X_train_selected, y_train, X_val_selected, y_val, X_test_selected, y_test
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test, etapa, modelo_nome):
        """Avaliar modelo e adicionar à tabela de performance"""
        model.fit(X_train, y_train)
        
        # Previsões
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Métricas
        f1_train = f1_score(y_train, y_pred_train)
        f1_val = f1_score(y_val, y_pred_val)
        f1_test = f1_score(y_test, y_pred_test)
        
        # Calcular melhoria
        if self.performance_table.empty:
            melhoria = "Baseline"
        else:
            ultimo_f1 = self.performance_table['F1_Teste'].iloc[-1]
            melhoria = f1_test - ultimo_f1
        
        # Adicionar à tabela
        new_row = {
            'Etapa': etapa,
            'Modelo': modelo_nome,
            'F1_Treino': f1_train,
            'F1_Validacao': f1_val,
            'F1_Teste': f1_test,
            'Melhoria': melhoria
        }
        
        self.performance_table = pd.concat([self.performance_table, pd.DataFrame([new_row])], ignore_index=True)
        
        print(f"\n📈 {etapa} - {modelo_nome}:")
        print(f"   F1 Treino: {f1_train:.4f}")
        print(f"   F1 Validação: {f1_val:.4f}")
        print(f"   F1 Teste: {f1_test:.4f}")
        print(f"   Melhoria: {melhoria:.4f}" if isinstance(melhoria, float) else f"   Melhoria: {melhoria}")
        
        return f1_test
    
    def plot_performance_evolution(self):
        """Plotar gráfico de evolução da performance"""
        plt.figure(figsize=(12, 8))
        
        # Gráfico de linhas
        etapas = self.performance_table['Etapa'] + ' - ' + self.performance_table['Modelo']
        plt.plot(etapas, self.performance_table['F1_Teste'], marker='o', linewidth=3, markersize=8, label='Teste')
        plt.plot(etapas, self.performance_table['F1_Treino'], marker='s', linestyle='--', markersize=6, label='Treino')
        plt.plot(etapas, self.performance_table['F1_Validacao'], marker='^', linestyle='-.', markersize=6, label='Validação')
        
        plt.title('Evolução do F1 Score durante o Projeto', fontsize=16, fontweight='bold')
        plt.xlabel('Etapas do Projeto', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('C:projeto_completo/results/performance_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Mostrar tabela
        print("\n📊 TABELA DE PERFORMANCE:")
        print("="*80)
        print(self.performance_table.round(4))
    
    def run_complete_ml_pipeline(self):
        """Executar pipeline completo com múltiplos modelos"""
        print("🎯 INICIANDO PROJETO COMPLETO DE MACHINE LEARNING")
        print("="*70)
        
        try:
            # 1. Carregar dados
            self.load_data()
            
            # 2. Pré-processar
            X_train, y_train, X_val, y_val, X_test, y_test = self.preprocess_data()
            
            # 3. Diferentes modelos e técnicas
            modelos = [
                {
                    'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                    'etapa': 'Baseline',
                    'nome': 'RandomForest'
                },
                {
                    'model': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
                    'etapa': 'Balanceamento',
                    'nome': 'RandomForest Balanced'
                },
                {
                    'model': RandomForestClassifier(
                        random_state=42, n_jobs=-1, class_weight='balanced',
                        max_depth=15, min_samples_split=10, min_samples_leaf=5
                    ),
                    'etapa': 'Pruning',
                    'nome': 'RandomForest Tunned'
                },
                {
                    'model': GradientBoostingClassifier(random_state=42, n_estimators=100),
                    'etapa': 'GradientBoosting',
                    'nome': 'GradientBoosting'
                },
                {
                    'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                    'etapa': 'XGBoost',
                    'nome': 'XGBoost'
                },
                {
                    'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                    'etapa': 'LogisticRegression',
                    'nome': 'Logistic Regression'
                }
            ]
            
            # 4. Treinar e avaliar cada modelo
            for config in modelos:
                self.evaluate_model(
                    config['model'], X_train, y_train, X_val, y_val, X_test, y_test,
                    config['etapa'], config['nome']
                )
            
            # 5. Ensemble dos melhores modelos
            from sklearn.ensemble import VotingClassifier
            best_rf = RandomForestClassifier(
                random_state=42, n_jobs=-1, class_weight='balanced',
                max_depth=15, min_samples_split=10, min_samples_leaf=5
            )
            best_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            
            ensemble = VotingClassifier(
                estimators=[('rf', best_rf), ('xgb', best_xgb)],
                voting='soft'
            )
            
            self.evaluate_model(
                ensemble, X_train, y_train, X_val, y_val, X_test, y_test,
                'Ensemble', 'Voting (RF + XGB)'
            )
            
            # 6. Plotar evolução
            self.plot_performance_evolution()
            
            # 7. Salvar melhor modelo
            self.model = ensemble  # Salvar o ensemble como melhor modelo
            self.save_models()
            
            print("\n✅ PROJETO CONCLUÍDO COM SUCESSO!")
            
        except Exception as e:
            print(f"❌ Erro no projeto: {e}")
            import traceback
            traceback.print_exc()
    
    def save_models(self):
        """Salvar modelos treinados"""
        import os
        os.makedirs('C:projeto_completo/models', exist_ok=True)
        os.makedirs('C:projeto_completo/results', exist_ok=True)
        
        joblib.dump(self.model, 'C:projeto_completo/models/best_model.pkl')
        joblib.dump(self.preprocessor, 'C:projeto_completo/models/preprocessor.pkl')
        joblib.dump(self.selector, 'C:projeto_completo/models/feature_selector.pkl')
        
        # Salvar tabela de performance
        self.performance_table.to_csv('C:projeto_completo/results/performance_table.csv', index=False)
        
        print("💾 Modelos e resultados salvos!")

# Executar projeto completo
if __name__ == "__main__":
    ml_project = AdultIncomeML()
    ml_project.run_complete_ml_pipeline()