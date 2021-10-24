import numpy as np


def convert2lower_triangular_matrix(a, b):
    """
    NxN 행렬을 입력받으면 하삼각행렬으로 변환해줌
    :param a: ax=b 일때 NxN 크기의 행렬
    :param b: ax=b 일때 b 값
    :return: a를 하삼각행렬로 변환해줌,b 도 따로 계산해서 리턴
    """
    n = len(b)
    # print(a)
    # 피벗행이 n-1 부터 1까지
    for k in range(n - 1, 0, -1):
        for i in range(k - 1, -1, -1):
            # print(k, i)
            # 단위행렬에 0 이 있으면 패스
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                # print(a[i, k], a[k, k])
                # print(a[i, :], "-", lam, a[k, :])
                a[i, :] = a[i, :] - lam * a[k, :]
                # print(a[i, :])
                # print(b[i], lam, b[k])
                b[i] = b[i] - lam * b[k]


    return a, b


def forward_substitution(a, b):
    """
    전입 대입 단계
    :param a: Ax = b 일때 A 값 N*N 계수행렬
    :param b: Ax = b 일때 b 값
    :return: b값 리턴
    """
    n = len(a)
    # 하삼각행렬에서 1행의 값은 이미 정해져있기때문에 바로 구함
    lam = b[0]
    ans = np.zeros(n)
    ans[0] = a[0, 0] / b[0]
    a[0, :] /= b[0]
    # 1행부터 채워나가면서 구함
    for i in range(1, n):
        temp_sum = 0
        for j in range(0, i):
            temp_sum += a[i, j] * ans[j]

        ans[i] = b[i]
        ans[i] = (ans[i] - temp_sum) / a[i, i]
    return ans


if __name__ == "__main__":
    # 책 예제
    q1 = """
    4x -2y + 1z = 11
    -2x + 4y -2z = -16
    1x -2y + 4z = 17
    
    x=1 y = -2 z =3
    """
    a3 = np.array([
        [4, -2, 1],
        [-2, 4, -2],
        [1, -2, 4]
    ], dtype=np.float64)
    b3 = np.array([
        11,
        -16,
        17
    ], dtype=np.float64)
    # 임의로 미지수가 5개인 연립 일차방정식
    q2 = """
    4x + y + 2z + 4a + 2b = 2
    x + y + z + a + b = 3
    3x + -2y + 3z + 2a + b = 2
    2x -y - z + 4a + 2b = -1
    1x + 2y+ 2z + 4a + -z
    
    x=1 y=2 z=-1 a=-2 b=3
    """
    a5 = np.array([
        [4, 1, 2, 4, 2],
        [1, 1, 1, 1, 1],
        [3, -2, 3, 2, 1],
        [2, -1, -1, 4, 2],
        [1,2,2,4,-1]
    ], dtype=np.float64)
    b5 = np.array([
        2, 3, -5, -1,-8
    ], dtype=np.float64)
    print(q1)
    ltmA, ltmB = convert2lower_triangular_matrix(a3, b3)
    result = forward_substitution(ltmA, ltmB)
    print(result)
    print(q2)
    ltmA, ltmB = convert2lower_triangular_matrix(a5, b5)
    result = forward_substitution(ltmA, ltmB)
    print(result)
