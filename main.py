import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from skfuzzy import control as ctrl

# Definição das variáveis linguísticas de ENTRADA e de SAÍDA
glicose = ctrl.Antecedent(np.arange(0, 200, 1), 'glicose')
imc = ctrl.Antecedent(np.arange(15, 40, 1), 'imc')
historico_familiar = ctrl.Antecedent(np.arange(0, 10, 1), 'historico_familiar')
diagnostico = ctrl.Consequent(np.arange(0, 100, 1), 'diagnostico')

# Definição dos Conjuntos Difusos (membership functions) para cada variável
glicose['baixa'] = fuzz.trimf(glicose.universe, [0, 0, 100])
glicose['normal'] = fuzz.trimf(glicose.universe, [0, 100, 200])
glicose['alta'] = fuzz.trimf(glicose.universe, [100, 200, 200])

imc['baixo'] = fuzz.trimf(imc.universe, [15, 15, 27])
imc['normal'] = fuzz.trimf(imc.universe, [15, 27, 40])
imc['alto'] = fuzz.trimf(imc.universe, [25, 40, 40])

historico_familiar['fraco'] = fuzz.trimf(historico_familiar.universe, [0, 0, 5])
historico_familiar['moderado'] = fuzz.trimf(historico_familiar.universe, [0, 5, 10])
historico_familiar['forte'] = fuzz.trimf(historico_familiar.universe, [5, 10, 10])

diagnostico['normal'] = fuzz.trimf(diagnostico.universe, [0, 0, 50])
diagnostico['pre_diabetes'] = fuzz.trimf(diagnostico.universe, [30, 50, 75]) 
diagnostico['diabetes_tipo_2'] = fuzz.trimf(diagnostico.universe, [75, 100, 100])



# Definição das Regras (Heurística Usada no processo de Raciocínio Fuzzy)
regra1 = ctrl.Rule(glicose['alta'] & imc['alto'] & historico_familiar['forte'], diagnostico['diabetes_tipo_2'])
regra2 = ctrl.Rule(glicose['normal'] & imc['normal'] & historico_familiar['fraco'], diagnostico['normal'])
regra3 = ctrl.Rule(glicose['baixa'] & imc['baixo'] & historico_familiar['moderado'], diagnostico['pre_diabetes'])
regra4 = ctrl.Rule(glicose['alta'] & imc['baixo'] & historico_familiar['fraco'], diagnostico['diabetes_tipo_2'])
regra5 = ctrl.Rule(glicose['baixa'] & imc['alto'] & historico_familiar['moderado'], diagnostico['pre_diabetes'])
regra6 = ctrl.Rule(glicose['normal'] & imc['baixo'] & historico_familiar['fraco'], diagnostico['normal'])
regra7 = ctrl.Rule(glicose['alta'] & imc['normal'] & historico_familiar['moderado'], diagnostico['diabetes_tipo_2'])
regra8 = ctrl.Rule(glicose['baixa'] & imc['baixo'] & historico_familiar['fraco'], diagnostico['normal'])


# Criação do Sistema de Controle
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5, regra6, regra7, regra8])
resultado_diagnostico = ctrl.ControlSystemSimulation(sistema_controle)


# Entradas do Sistema de Diagnóstico (usuário fornece os valores)
glicose_input = float(input("Digite o valor de glicose: "))
imc_input = float(input("Digite o valor de IMC: "))
historico_familiar_input = float(input("Digite o valor do histórico familiar: "))

resultado_diagnostico.input['glicose'] = glicose_input
resultado_diagnostico.input['imc'] = imc_input
resultado_diagnostico.input['historico_familiar'] = historico_familiar_input

# Processo de Inferência
resultado_diagnostico.compute()

# Saída do Sistema de Diagnóstico
valor_diagnostico = resultado_diagnostico.output['diagnostico']
print("Valor de saída defuzzificado:", valor_diagnostico)

# Interpretação do Diagnóstico
if valor_diagnostico <= 50:
    print("Diagnóstico: Normal")
elif 50 < valor_diagnostico <= 75:
    print("Diagnóstico: Pré-diabetes")
else:
    print("Diagnóstico: Diabetes Tipo 2")

# Visualização dos Conjuntos Difusos
diagnostico.view(sim=resultado_diagnostico)
plt.show()
