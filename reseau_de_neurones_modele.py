import numpy as np
import random
import copy
from collections import deque

## paramètres
NB_PARTIES = 20000
NB_SAMPLES_MAX = 50000
p_debut = 1
p_fin = 0.01
NB_SAMPLES_DEBUT_ENTRAINEMENT = 5000
TAILLE_BATCHS = 128
NB_ENTRAINEMENT_BATCH = 2
gamma = 0.95
learning_rate = 0.0001
ACTU_W_TARGET = 1000  # tous les ... batchs
PERIODE_STOCKAGE_PC = 4800000 // ACTU_W_TARGET  # toutes les ... maj du reseau
NB_NEURONES_LAYER1 = 128
NB_NEURONES_LAYER2 = 64
TAILLE_STATE = 10
NB_ACTIONS_POSSIBLES = 4  # on suppose les actions numérotées 1, 2, ... NB_ACTIONS_POSSIBLES
## paramètres

'''
state ():
actions ():
rewards:
sample = [state1, state2, action, reward, terminal_state] ()
'''
TAILLE_SAMPLE = 2 * TAILLE_STATE + 3

try:
    data = np.load('reseau_neurones.npz')
    W1 = data['W1']
    W2 = data['W2']
    W3 = data['W3']
    B1 = data['B1']
    B2 = data['B2']
    B3 = data['B3']
except FileNotFoundError:
    W1 = np.random.randn(NB_NEURONES_LAYER1, TAILLE_STATE) * np.sqrt(2 / TAILLE_STATE)  # He init pour ReLU
    W2 = np.random.randn(NB_NEURONES_LAYER2, NB_NEURONES_LAYER1) * np.sqrt(2 / NB_NEURONES_LAYER1)
    W3 = np.random.randn(NB_ACTIONS_POSSIBLES, NB_NEURONES_LAYER2) * np.sqrt(2 / NB_NEURONES_LAYER2)
    B1 = np.zeros(NB_NEURONES_LAYER1)
    B2 = np.zeros(NB_NEURONES_LAYER2)
    B3 = np.zeros(NB_ACTIONS_POSSIBLES)

samples = np.zeros((NB_SAMPLES_MAX, TAILLE_SAMPLE), dtype=np.float32)  # 300+300+3
samples_count = 0  # Nombre réel de samples stockés
head = 0  # Index circulaire (tête)


def build_state():
    state = np.zeros(TAILLE_STATE, dtype=np.float32)

    pass

    return state


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


W1_target, W2_target, W3_target = W1.copy(), W2.copy(), W3.copy()
B1_target, B2_target, B3_target = B1.copy(), B2.copy(), B3.copy()
compteur_target = 0
for partie in range(NB_PARTIES):
    p = p_debut - (p_debut - p_fin) * partie / NB_PARTIES

    if partie % 100 == 0:
        print(f"progression : {partie * 100 / NB_PARTIES} %")

    ## initialisation de la partie
    pass
    ##

    partie_en_cours = True
    while partie_en_cours:
        state1 = build_state()
        ## choix d'une action
        if random.random() <= p:
            action = random.randint(1, NB_ACTIONS_POSSIBLES)
        else:
            A0 = np.array(state1)
            Z1 = np.dot(W1, A0) + B1
            A1 = relu(Z1)
            Z2 = np.dot(W2, A1) + B2
            A2 = relu(Z2)
            Z3 = np.dot(W3, A2) + B3
            A3 = Z3
            action = np.argmax(A3) + 1

        reward = 0
        ## on execute l'action
        pass
        ##

        terminal_state = not partie_en_cours
        state2 = build_state()
        sample = np.concatenate([
            state1,
            state2,
            [action, reward, terminal_state]
        ])
        samples[head] = sample
        head = (head + 1) % len(samples)
        if samples_count < len(samples):
            samples_count += 1

        if samples_count >= NB_SAMPLES_DEBUT_ENTRAINEMENT:
            for _ in range(NB_ENTRAINEMENT_BATCH):
                indices = np.random.choice(samples_count, TAILLE_BATCHS, replace=False)
                selection = samples[indices]  # (100, 603)

                # --- 1. Extraction des données du batch (vectorisé) ---
                states1_batch = np.array([sample[:TAILLE_STATE] for sample in selection])
                states2_batch = np.array([sample[TAILLE_STATE:2*TAILLE_STATE] for sample in selection])
                actions_batch = np.array([int(sample[2*TAILLE_STATE]) for sample in selection])
                rewards_batch = np.array([sample[2*TAILLE_STATE+1] for sample in selection])
                terminal_states_batch = np.array([bool(sample[TAILLE_STATE+2]) for sample in selection])

                # --- 2. Forward Pass pour Q_target (réseau cible) ---
                Q_target = np.zeros(TAILLE_BATCHS)
                non_terminal_mask = ~terminal_states_batch
                Z1 = states2_batch @ W1.T + B1
                A1 = relu(Z1)
                Z2 = A1 @ W2.T + B2
                A2 = relu(Z2)
                Z3 = A2 @ W3.T + B3
                if np.any(non_terminal_mask):
                    Z1_target = states2_batch @ W1_target.T + B1_target
                    A1_target = relu(Z1_target)
                    Z2_target = A1_target @ W2_target.T + B2_target
                    A2_target = relu(Z2_target)
                    Z3_target = A2_target @ W3_target.T + B3_target
                    best_actions = np.argmax(Z3, axis=1)
                    Q_target[non_terminal_mask] = (rewards_batch[non_terminal_mask]
                        + gamma * Z3_target[non_terminal_mask, best_actions[non_terminal_mask]])
                Q_target[terminal_states_batch] = rewards_batch[terminal_states_batch]

                # --- 3. Forward Pass pour Q (réseau principal) ---
                Z1 = states1_batch @ W1.T + B1
                A1 = relu(Z1)
                Z2 = A1 @ W2.T + B2
                A2 = relu(Z2)
                Z3 = A2 @ W3.T + B3
                Q = Z3[np.arange(TAILLE_BATCHS), actions_batch - 1]  # Extraction des Q pour chaque action

                # --- 4. Backward Pass (vectorisé) ---
                gradientaC = Q - Q_target

                delta3 = np.zeros((TAILLE_BATCHS, NB_ACTIONS_POSSIBLES))
                delta3[np.arange(TAILLE_BATCHS), actions_batch - 1] = gradientaC

                delta2 = (delta3 @ W3) * relu_derivative(Z2)
                delta1 = (delta2 @ W2) * relu_derivative(Z1)

                # Gradients pour W1, W2, B1, B2
                dW1 = delta1.T @ states1_batch
                dW2 = delta2.T @ A1
                dW3 = delta3.T @ A2
                dB1 = np.sum(delta1, axis=0)
                dB2 = np.sum(delta2, axis=0)
                dB3 = np.sum(delta3, axis=0)

                # --- 5. Mise à jour des poids ---
                W1 -= learning_rate * dW1 / TAILLE_BATCHS
                W2 -= learning_rate * dW2 / TAILLE_BATCHS
                W3 -= learning_rate * dW3 / TAILLE_BATCHS
                B1 -= learning_rate * dB1 / TAILLE_BATCHS
                B2 -= learning_rate * dB2 / TAILLE_BATCHS
                B3 -= learning_rate * dB3 / TAILLE_BATCHS

                # Mise à jour du réseau cible
                compteur_target += 1
                if compteur_target % ACTU_W_TARGET == 0:
                    W1_target, W2_target, W3_target = W1.copy(), W2.copy(), W3.copy()
                    B1_target, B2_target, B3_target = B1.copy(), B2.copy(), B3.copy()


                if compteur_target % PERIODE_STOCKAGE_PC == 0:
                    np.savez("reseau_neurones.npz", W1=W1, W2=W2, W3=W3, B1=B1, B2=B2, B3=B3)
                    print(f"partie : {partie}   Poids, biais exportés dans reseau_neurones.npz")

np.savez("reseau_neurones.npz", W1=W1, W2=W2, W3=W3, B1=B1, B2=B2, B3=B3)
print("Poids, biais exportés dans reseau_neurones.npz")