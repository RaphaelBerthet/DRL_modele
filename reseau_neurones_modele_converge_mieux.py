import numpy as np
import random

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
ACTU_W_TARGET = 1000  # tous les ... majs du reseau
PERIODE_STOCKAGE_PC = 4800000 // ACTU_W_TARGET  # toutes les ... majs du reseau
NB_NEURONES_LAYER1 = 128
NB_NEURONES_LAYER2 = 64
TAILLE_STATE = 10
NB_ACTIONS_POSSIBLES = 4  # on suppose les actions numérotées 1, 2, ... NB_ACTIONS_POSSIBLES
MAX_NORME_GRADIENT = 10  # norme 2
DELTA_HUBER_LOSS = 1
beta1, beta2, eps = 0.9, 0.999, 1e-8
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
    mW1 = data['mW1']; vW1 = data['vW1']
    mW2 = data['mW2']; vW2 = data['vW2']
    mW3 = data['mW3']; vW3 = data['vW3']
    mB1 = data['mB1']; vB1 = data['vB1']
    mB2 = data['mB2']; vB2 = data['vB2']
    mB3 = data['mB3']; vB3 = data['vB3']
    t_adam = int(data['t_adam'])

except FileNotFoundError:
    W1 = np.random.randn(NB_NEURONES_LAYER1, TAILLE_STATE) * np.sqrt(2 / TAILLE_STATE)  # He init pour ReLU
    W2 = np.random.randn(NB_NEURONES_LAYER2, NB_NEURONES_LAYER1) * np.sqrt(2 / NB_NEURONES_LAYER1)
    W3 = np.random.randn(NB_ACTIONS_POSSIBLES, NB_NEURONES_LAYER2) * np.sqrt(2 / NB_NEURONES_LAYER2)
    B1 = np.zeros(NB_NEURONES_LAYER1)
    B2 = np.zeros(NB_NEURONES_LAYER2)
    B3 = np.zeros(NB_ACTIONS_POSSIBLES)
    mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
    mW3 = np.zeros_like(W3); vW3 = np.zeros_like(W3)
    mB1 = np.zeros_like(B1); vB1 = np.zeros_like(B1)
    mB2 = np.zeros_like(B2); vB2 = np.zeros_like(B2)
    mB3 = np.zeros_like(B3); vB3 = np.zeros_like(B3)
    t_adam = 0

samples = np.zeros((NB_SAMPLES_MAX, TAILLE_SAMPLE), dtype=np.float32)  # 300+300+3
samples_count = 0  # Nombre réel de samples stockés
head = 0  # Index circulaire (tête)

def build_state():
    state = np.zeros(TAILLE_STATE, dtype=np.float32)

    pass

    return state

def adam_update(param, grad, m, v, t, lr):
    m[:] = beta1 * m + (1 - beta1) * grad
    v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


W1_target, W2_target, W3_target = W1.copy(), W2.copy(), W3.copy()
B1_target, B2_target, B3_target = B1.copy(), B2.copy(), B3.copy()
ct_majs_reseau = 0

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
                selection = samples[indices]

                # --- 1. Extraction des données du batch (vectorisé) ---
                states1_batch = selection[:, :TAILLE_STATE]
                states2_batch = selection[:, TAILLE_STATE:2*TAILLE_STATE]
                actions_batch = selection[:, 2*TAILLE_STATE].astype(int)
                rewards_batch = selection[:, 2*TAILLE_STATE+1]
                terminal_states_batch = selection[:, 2*TAILLE_STATE+2].astype(bool)

                # --- 2. Forward Pass pour Q_target (réseau cible) ---
                Q_target = np.zeros(TAILLE_BATCHS)
                non_terminal_mask = ~terminal_states_batch
                Z1_s2 = states2_batch @ W1.T + B1
                A1_s2 = relu(Z1_s2)
                Z2_s2 = A1_s2 @ W2.T + B2
                A2_s2 = relu(Z2_s2)
                Z3_s2 = A2_s2 @ W3.T + B3
                if np.any(non_terminal_mask):
                    Z1_target = states2_batch @ W1_target.T + B1_target
                    A1_target = relu(Z1_target)
                    Z2_target = A1_target @ W2_target.T + B2_target
                    A2_target = relu(Z2_target)
                    Z3_target = A2_target @ W3_target.T + B3_target
                    best_actions = np.argmax(Z3_s2, axis=1)
                    Q_target[non_terminal_mask] = (rewards_batch[non_terminal_mask]
                        + gamma * Z3_target[non_terminal_mask, best_actions[non_terminal_mask]])
                Q_target[terminal_states_batch] = rewards_batch[terminal_states_batch]

                # --- 3. Forward Pass pour Q (réseau principal) ---
                Z1_s1 = states1_batch @ W1.T + B1
                A1_s1 = relu(Z1_s1)
                Z2_s1 = A1_s1 @ W2.T + B2
                A2_s1 = relu(Z2_s1)
                Z3_s1 = A2_s1 @ W3.T + B3
                Q = Z3_s1[np.arange(TAILLE_BATCHS), actions_batch - 1]  # Extraction des Q pour chaque action

                # --- 4. Backward Pass (vectorisé) ---
                erreur = Q - Q_target

                gradientaC = np.where(
                    np.abs(erreur) <= DELTA_HUBER_LOSS,
                    erreur,
                    DELTA_HUBER_LOSS * np.sign(erreur)
                )

                delta3 = np.zeros((TAILLE_BATCHS, NB_ACTIONS_POSSIBLES))
                delta3[np.arange(TAILLE_BATCHS), actions_batch - 1] = gradientaC

                delta2 = (delta3 @ W3) * relu_derivative(Z2_s1)
                delta1 = (delta2 @ W2) * relu_derivative(Z1_s1)

                # Gradients pour W1, W2, B1, B2
                dW1 = delta1.T @ states1_batch
                dW2 = delta2.T @ A1_s1
                dW3 = delta3.T @ A2_s1
                dB1 = np.sum(delta1, axis=0)
                dB2 = np.sum(delta2, axis=0)
                dB3 = np.sum(delta3, axis=0)
                norme = np.sqrt(
                    np.sum(dW1**2) +
                    np.sum(dW2**2) +
                    np.sum(dW3**2) +
                    np.sum(dB1**2) +
                    np.sum(dB2**2) +
                    np.sum(dB3**2)
                )
                if norme > MAX_NORME_GRADIENT:
                    facteur = MAX_NORME_GRADIENT / norme
                    dW1 *= facteur
                    dW2 *= facteur
                    dW3 *= facteur
                    dB1 *= facteur
                    dB2 *= facteur
                    dB3 *= facteur
                
                # --- 5. Mise à jour des poids ---
                t_adam += 1
                adam_update(W1, dW1 / TAILLE_BATCHS, mW1, vW1, t_adam, learning_rate)
                adam_update(W2, dW2 / TAILLE_BATCHS, mW2, vW2, t_adam, learning_rate)
                adam_update(W3, dW3 / TAILLE_BATCHS, mW3, vW3, t_adam, learning_rate)
                adam_update(B1, dB1 / TAILLE_BATCHS, mB1, vB1, t_adam, learning_rate)
                adam_update(B2, dB2 / TAILLE_BATCHS, mB2, vB2, t_adam, learning_rate)
                adam_update(B3, dB3 / TAILLE_BATCHS, mB3, vB3, t_adam, learning_rate)

                # Mise à jour du réseau cible
                ct_majs_reseau += 1
                if ct_majs_reseau % ACTU_W_TARGET == 0:
                    W1_target, W2_target, W3_target = W1.copy(), W2.copy(), W3.copy()
                    B1_target, B2_target, B3_target = B1.copy(), B2.copy(), B3.copy()


                if ct_majs_reseau % PERIODE_STOCKAGE_PC == 0:
                    np.savez("reseau_neurones.npz", W1=W1, W2=W2, W3=W3, B1=B1, B2=B2, B3=B3, mW1=mW1, mW2=mW2, mW3=mW3, mB1=mB1, mB2=mB2, mB3=mB3, vW1=vW1, vW2=vW2, vW3=vW3, vB1=vB1, vB2=vB2, vB3=vB3, t_adam=t_adam)
                    print(f"partie : {partie}   Poids, biais exportés dans reseau_neurones.npz")

np.savez("reseau_neurones.npz", W1=W1, W2=W2, W3=W3, B1=B1, B2=B2, B3=B3, mW1=mW1, mW2=mW2, mW3=mW3, mB1=mB1, mB2=mB2, mB3=mB3, vW1=vW1, vW2=vW2, vW3=vW3, vB1=vB1, vB2=vB2, vB3=vB3, t_adam=t_adam)
print("Poids, biais exportés dans reseau_neurones.npz")