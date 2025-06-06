# Sistema de Previsão de Ações Brasileiras com Redes Neurais LSTM

"+ Inicialização do front-end feita por meio da bibliotec streamlit, facilitando o deploy +"
                  
 ↓ " run streamlit sistema_previsao.py" no terminal
 
"+ Verificação do treinamento do modelo LSTM - no tópico 3 em "sistema_previsao_notebook.ipynb" +"
... 

Este sistema fullstack implementa um modelo completo de previsão para as ações BRL
- **Vivara (VIVA3)** com LSTM próprio
- **Movida (MOVI3)** com LSTM próprio
- **Trisul (TRIS3)** com LSTM próprio
- **Magazine Luiza (MGLU3)** sem LSTM próprio
- **Americanas (AMER3)** sem LSTM próprio

## Funcionalidades:
- Coleta de dados históricos dos últimos 5 anos
- Análise técnica completa
- Modelo LSTM para previsão de preços
- Detecção de eventos significativos (grandes altas/baixas)
- Recomendações de **COMPRAR**, **VENDER** ou **MANTER**
- Níveis de probabilidade baseados no histórico
- Análise para diferentes horizontes temporais (1 semana, 1 mês, 3 meses, 6 meses, 1 ano)

- ## Necessidade de correção de bugs:
- Erros de exibição, sendo os prinipais de previsão estimada de preços futuros e aba do Machine Learning
- Não atualização correta das datas, entretanto quando atualizados os valores de preços atuais, ainda funciona corretamente o sistema de previsão de valorespor meio do uso da biblioteca: TensorFlow

