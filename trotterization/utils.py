import numpy as np
from qulacs import Observable, PauliOperator, QuantumCircuit
from qulacs.gate import (
    CNOT,
    RX,
    RY,
    RZ,
    DenseMatrix,
    H,
    PauliRotation,
    S,
    Sdag,
    X,
    Z,
    add,
    merge,
)


def apply_pauli_gates(gate_lst, qubit_indices, pauli_ids, right_side=False):
    """指定された量子ビットに対してPauliゲートを適用"""
    # print(qubit_indices, pauli_ids)
    for idx, pauli_id in zip(qubit_indices, pauli_ids):
        if pauli_id == 1:  # X
            gate_lst.append(H(idx))
        elif pauli_id == 2:  # Y
            if right_side:
                gate_lst.append(H(idx))
                gate_lst.append(S(idx))
            else:
                gate_lst.append(Sdag(idx))
                gate_lst.append(H(idx))


def gen_pauli_rotation_gate(qubit_indices: list, pauli_ids: list, angle: float, with_cnot: bool = False) -> list:
    gate_lst = []
    if with_cnot:
        position = max(qubit_indices)
        apply_pauli_gates(gate_lst, qubit_indices, pauli_ids, right_side=False)
        if len(qubit_indices) >= 2:  # CNOT必要
            # CNOTゲートを適用
            for idx in qubit_indices:
                if idx != position:
                    gate_lst.append(CNOT(idx, position))
        gate_lst.append(RZ(position, -1 * angle))  # qulacsでは回転角が逆で定義されている
        if len(qubit_indices) >= 2:  # CNOT必要
            # CNOTゲートを適用
            for idx in qubit_indices:
                if idx != position:
                    gate_lst.append(CNOT(idx, position))
        apply_pauli_gates(gate_lst, qubit_indices, pauli_ids, right_side=True)
        return gate_lst
    else:
        gate_lst = [PauliRotation(qubit_indices, pauli_ids, -1 * angle)]  # qulacsでは回転角が逆で定義されている
        return gate_lst


def gen_observable(nqubits):
    # generate observable <O> = 1/n \sum_i^n Z_i
    obs = Observable(nqubits)
    for i in range(nqubits):
        obs.add_operator(PauliOperator("Z " + str(i), 1.0 / nqubits))
    return obs


def calculate_exact_time_evolution_unitary_matrix(nqubits, coef_J, coef_h, delta):
    # e^{-iHt}を直接対角化する。Hの行列表現を得るために、gateを生成してそのmatrixを取得する
    zz_matrix = coef_J * np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  ## Z_i*Z_{i+1}の行列表示
    hx_matrix = coef_h * np.array([[0, 1], [1, 0]])
    zz = DenseMatrix([0, 1], zz_matrix)  ## 0~1間の相互作用
    hx = DenseMatrix(0, hx_matrix)  ## 0サイトへの横磁場
    ## qulacs.gate.addを用いて、1以降のサイトの相互作用と横磁場を足していく
    for i in range(1, nqubits):
        zz = add(zz, DenseMatrix([i, (i + 1) % nqubits], zz_matrix))
        hx = add(hx, DenseMatrix(i, hx_matrix))
    ## 最終的なハミルトニアン
    ham = add(zz, hx)
    matrix = ham.get_matrix()  # 行列の取得
    eigenvalue, P = np.linalg.eigh(np.array(matrix))  # 取得した行列の固有値、固有ベクトルを取得
    ## e^{-i*H*delta}を行列として作る
    e_iHdelta = np.diag(np.exp(-1.0j * eigenvalue * delta))
    e_iHdelta = np.dot(P, np.dot(e_iHdelta, P.T))
    return e_iHdelta


def generate_one_trotter_layer_circuit(nqubits, coef_J, coef_h, delta, order=1, use_cnots=False):
    one_layer_circuit = QuantumCircuit(nqubits)

    time_ZZ = 2 * delta * coef_J
    time_X = 2 * delta * coef_h

    if order == 1:
        ### 1st order Trotter decomposition(U_1(t) = e^{t/n A} e^{t/n B})
        gates = []
        for i in range(nqubits):
            gates += gen_pauli_rotation_gate([i, (i + 1) % nqubits], [3, 3], time_ZZ, with_cnot=use_cnots)
            gates.append(RX(i, time_X))

    elif order == 2:
        ### 2nd order Trotter decomposition(U_2(t) = e^{t/2n A} e^{t/n B} e^{t/2n A})
        gates = []
        for i in range(nqubits):
            zz_elements = gen_pauli_rotation_gate(
                [i, (i + 1) % nqubits], [3, 3], time_ZZ / 2, with_cnot=use_cnots
            )  # angle changed! (time_ZZ -> time_ZZ / 2)
            gates += zz_elements
            gates.append(RX(i, time_X))
            gates += zz_elements

    elif order == 4:
        ### 4th order Trotter decomposition
        gates = []
        val_s_2 = 1 / (4 - np.cbrt(4))
        u_2_elements = []
        for _ in range(2):  # repeat U_2(s_2 t) twice
            for i in range(nqubits):
                zz_elements = gen_pauli_rotation_gate(
                    [i, (i + 1) % nqubits], [3, 3], val_s_2 * time_ZZ / 2, with_cnot=use_cnots
                )  # angle changed! (time_ZZ -> time_ZZ / 2)
                u_2_elements += zz_elements
                u_2_elements.append(RX(i, val_s_2 * time_X))
                u_2_elements += zz_elements
        gates += u_2_elements

        for i in range(nqubits):  # gen U_2((1-4*s_2) t)
            zz_elements = gen_pauli_rotation_gate(
                [i, (i + 1) % nqubits], [3, 3], (1 - 4 * val_s_2) * time_ZZ / 2, with_cnot=use_cnots
            )
            gates += zz_elements
            gates.append(RX(i, (1 - 4 * val_s_2) * time_X))
            gates += zz_elements

        # add two U_2(s_2 t) again
        gates += u_2_elements
    else:
        raise ValueError("order should be 1 or 2 or 4")

    # apply gates to the circuit
    for gate in gates:
        one_layer_circuit.add_gate(gate)
    return one_layer_circuit


def simulate_trotter_dynamics(one_layer_circuit, initial_state, observable, n_steps):
    state = initial_state.copy()
    value_lst = []
    # t=0
    value_lst.append(observable.get_expectation_value(state))

    # t=0以降の全磁化を計算
    for _ in range(n_steps):
        # delta=t/Mだけ時間発展
        one_layer_circuit.update_quantum_state(state)
        # 磁化を計算して記録
        value_lst.append(observable.get_expectation_value(state))
    return value_lst
