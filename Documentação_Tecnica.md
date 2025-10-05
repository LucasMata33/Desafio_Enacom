# Documentação Técnica - Decisões e Resultados

## Informações do Projeto

**Projeto:** Sistema de Previsão de Preços de Energia Elétrica  
**Submercado:** Sudeste  
**Período dos dados:** 2012-01-01 a 2025-09-14  
**Total de registros:** 716 (dados semanais)  
**Data de execução:** 2025  

---

## 1. Decisões de Arquitetura

### 1.1 Estrutura do Projeto

**Decisão:** estrutura original fornecida no desafio

**Justificativa:**
- Compatibilidade com testes existentes
- Reduz risco de quebrar funcionalidades

**Arquivos criados:**
- `improved_predictor.py` - Modelo ensemble
- `main_comparison.py` - Script de comparação
- `generate_report.py` - Geração de relatórios
- `loaders/newave_loader.py` - Atualizado

**Arquivos mantidos sem alteração:**
- `predictor/next_step_predictor.py` - Baseline
- `loaders/csv_loader.py` - Loader de preços
- `tests/*` - Suite de testes
- `main.py` - Script original

### 1.2 Abordagem de Modelagem

**Decisão:** Modelo ensemble com três componentes

**Componentes escolhidos:**
1. Análise de tendência (EMA) - 40%
2. Correlações entre maturidades - 35%
3. Sazonalidade mensal - 25%

**Justificativa:**
- EMA captura dinâmica recente do mercado
- Correlações aproveitam informação entre séries relacionadas
- Sazonalidade captura padrões anuais do setor elétrico

**Alternativas consideradas e descartadas:**
- ARIMA: complexidade alta, difícil interpretabilidade
- LSTM: requer mais dados, maior tempo de treinamento
- XGBoost: necessita feature engineering extensivo

---

## 2. Decisões de Implementação

### 2.1 Médias Móveis Exponenciais (EMA)

**Janelas escolhidas:** 4, 12, 26 semanas

**Justificativa:**
- 4 semanas: tendência de curtíssimo prazo (1 mês)
- 12 semanas: tendência de médio prazo (3 meses)
- 26 semanas: tendência de longo prazo (6 meses)

**Implementação:**
```python
ema_4 = data.ewm(span=4, adjust=False).mean()
ema_12 = data.ewm(span=12, adjust=False).mean()
ema_26 = data.ewm(span=26, adjust=False).mean()
```

**Alternativas consideradas:**
- SMA (Simple Moving Average): descartada por não dar peso maior a dados recentes
- Janelas maiores (52 semanas): descartada por reduzir responsividade

### 2.2 Correlações entre Maturidades

**Decisão:** Usar top-3 maturidades mais correlacionadas

**Justificativa:**
- Evita ruído de correlações fracas
- Reduz complexidade computacional
- Mantém foco em relações mais fortes

**Observação dos dados:**
- M+0 ↔ M+1: correlação ~0.95
- A+0 ↔ A+1: correlação ~0.98
- M+ ↔ A+: correlação 0.60-0.80

### 2.3 Decomposição Sazonal

**Decisão:** Agregação mensal simples

**Justificativa:**
- Padrões do setor elétrico são mensais (período seco/úmido)
- Implementação simples e interpretável
- Baixo risco de overfitting

**Padrão observado:**
- Agosto-Outubro: preços mais altos (período seco)
- Dezembro-Março: preços mais baixos (período úmido)

### 2.4 Pesos do Ensemble

**Decisão:** 40% tendência, 35% correlação, 25% sazonalidade

**Justificativa:**
- Tendência tem maior impacto em previsões de curto prazo
- Correlações são fortes mas não determinísticas
- Sazonalidade é relevante mas não dominante

**Método de definição:**
- Análise empírica dos dados históricos
- Testes com diferentes combinações
- Não houve otimização automática por grid search

---

## 3. Decisões de Validação

### 3.1 Split de Dados

**Decisão:** 80% treino, 20% teste (split temporal)

**Valores:**
- Treino: 572 registros (2012-01-01 a ~2023-07)
- Teste: 144 registros (~2023-07 a 2025-09-14)

**Justificativa:**
- Split temporal respeita natureza de séries temporais
- 20% de teste fornece amostra significativa (144 registros)
- Evita data leakage

**Alternativas consideradas:**
- Split aleatório: descartado (viola premissa temporal)
- K-fold: descartado (não apropriado para séries temporais)
- Walk-forward: ideal mas não implementado por limitação de tempo

### 3.2 Horizonte de Previsão

**Decisão:** 3 passos à frente

**Justificativa:**
- Horizonte prático para operações de trading
- Balanceia complexidade e utilidade
- Permite avaliação robusta

### 3.3 Métricas de Avaliação

**Métricas escolhidas:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

**Justificativa:**
- MAE: interpretável diretamente em R$/MWh
- RMSE: penaliza erros grandes
- MAPE: permite comparação entre maturidades

---

## 4. Implementação do NewaveLoader

### 4.1 Parser do Formato

**Desafio:** Arquivo cmarg*.out tem formato complexo

**Estrutura identificada:**
- Header com metadados
- Dados organizados por mês e patamar
- 3 patamares por mês (PAT 1, 2, 3)
- 12 cenários + coluna MEDIA

**Decisão de parsing:**
```python
# Agregar patamares calculando média
df_agg = df.groupby('date').agg({
    'scenario_1': 'mean',
    ...,
    'mean': 'mean'
})
```

**Justificativa:**
- Simplifica dados para uso no modelo
- Mantém informação essencial (custos médios)
- Reduz dimensionalidade

### 4.2 Limitação de Dados

**Situação:** Apenas deck de agosto/2025 disponível

**Decisão:** Implementar loader completo mas não integrar ao modelo

**Justificativa:**
- Modelo requer histórico completo (2012-2025)
- Integração seria trivial com dados disponíveis

**Plano de integração futura:**
1. Coletar histórico mensal do Newave
2. Alinhar temporalmente com preços semanais
3. Adicionar como feature com peso 15-20%
4. Avaliar melhoria incremental

---

## 5. Resultados Obtidos

### 5.1 Métricas Gerais

**Dataset de teste:** 144 registros (20.1%)  
**Horizonte de avaliação:** 3 passos à frente  

### 5.2 Maturidades de Longo Prazo (A+)

| Maturidade | Baseline MAPE | Melhorado MAPE | Melhoria |
|------------|---------------|----------------|----------|
| A + 0 | 6.24% | 5.49% | +11.9% |
| A + 1 | 4.49% | 2.46% | +45.1% |
| A + 2 | 3.90% | 2.12% | +45.5% |
| A + 3 | 2.22% | 0.96% | +56.9% |
| A + 4 | 1.66% | 0.52% | +68.6% |

**MAPE médio baseline:** 3.70%  
**MAPE médio melhorado:** 2.31%  
**Melhoria média:** +45.4%

### 5.3 Maturidades de Curto Prazo (M+)

| Maturidade | Baseline MAPE | Melhorado MAPE | Melhoria |
|------------|---------------|----------------|----------|
| M + 0 | 6.37% | 7.19% | -12.8% |
| M + 1 | 0.67% | 2.32% | -248.3% |
| M + 2 | 1.31% | 2.48% | -88.8% |
| M + 3 | 0.59% | 1.75% | -195.0% |

**MAPE médio baseline:** 2.24%  
**MAPE médio melhorado:** 3.44%  
**Melhoria média:** -121.2%

### 5.4 Resultado Consolidado

**Maturidades melhoradas:** 5/9 (55.6%)  
**Maturidades pioradas:** 4/9 (44.4%)

---

## 6. Análise dos Resultados

### 6.1 Por que o Ensemble Falhou em Curto Prazo

**Hipótese 1: Over-smoothing**

Alta volatilidade e comportamento errático, influenciado por fatores de curtíssimo prazo (clima, carga instantânea, indisponibilidade de geração).
Menor influência de tendências estruturais, que são justamente as que o ensemble tenta capturar.
O modelo baseline de persistência já é extremamente competitivo, pois no curto prazo o preço de amanhã tende a ser semelhante ao de hoje.
O ensemble, ao tentar aprender padrões mais complexos, acabou “superajustando” e perdendo eficiência frente a um modelo simples.

**Hipótese 2: Features adicionam ruído**

Em horizontes curtos, correlações e sazonalidade podem introduzir ruído em vez de sinal preditivo.

**Evidência:**
- M+1 teve pior desempenho (-248%)
- M+1 é altamente volátil mas tem correlação forte com M+0
- Correlação pode ter propagado ruído

**Hipótese 3: Período de teste atípico**

Os últimos 20% dos dados (teste) podem ter comportamento diferente do histórico de treino.

### 6.2 Por que o Ensemble Funcionou em Longo Prazo

**Fator 1: Estabilidade**

Maturidades A+ têm menor volatilidade, permitindo que features de tendência e sazonalidade capturem padrões reais.

**Fator 2: Correlações fortes**

Correlações entre maturidades A+ são muito altas (>0.95), permitindo propagação de informação com baixo ruído.

**Fator 3: Tendências confiáveis**

Horizontes longos têm tendências mais suaves e previsíveis, favorecendo análise EMA.

---

### 7 Dashboard Interativo

**Decisão:** Criar visualização HTML standalone
- Facilita apresentação visual
- Funciona offline
- Não requer dependências externas

**Funcionalidades implementadas:**
- 4 abas navegáveis (Visão Geral, Análise, Insights, Dados)
- Gráficos de barras comparativos
- Tabela completa de métricas
- Análise crítica dos resultados

---

## 8. Conclusão

### Objetivos Alcançados

✅ Modelo preditivo melhorado para longo prazo  
✅ NewaveLoader completo e funcional  
✅ Avaliação quantitativa rigorosa  
✅ Dashboard interativo para análise  
✅ Documentação completa de decisões  

### Objetivos Parciais

⚠️ Melhoria em curto prazo (falhou, mas entendido)  
⚠️ Integração Newave (código pronto, faltam dados históricos)  

### Contribuições Principais

**Técnica:**
- Implementação de modelo ensemble interpretável
- Parser completo do formato Newave
- Framework de comparação extensível

**Analítica:**
- Identificação clara de limitações por horizonte
- Proposta de modelo híbrido específico
- Roadmap de melhorias baseado em evidências

---

## Anexos

### A. Requisitos de Sistema

**Python:** 3.8+  
**Bibliotecas:**
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0

### B. Arquivos Entregues

1. `improved_predictor.py` - Modelo ensemble (500 linhas)
2. `main_comparison.py` - Script de comparação (200 linhas)
3. `generate_report.py` - Gerador de relatórios (300 linhas)
4. `loaders/newave_loader.py` - Parser Newave (250 linhas)
5. `dashboard.html` - Interface visual (300 linhas)
6. `DOCUMENTACAO_TECNICA.md` - Este documento

### C. Testes

**Suite original:** 35/35 testes passando  
**Cobertura:** 77% (484 linhas, 110 não cobertas)

### D. Referências

- CCEE: https://www.ccee.org.br
- ONS: https://www.ons.org.br
- Newave: Programa CEPEL de otimização energética
- Inewave: Biblioteca Python para manipulação Newave

---

**Documento gerado em:** 2025  
**Autor:** Lucas Galdino da Mata  
**Projeto:** Desafio de Modelagem Preditiva - Preços de Energia
