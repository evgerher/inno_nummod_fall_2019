import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix

def get_buildings():
  grid = np.zeros((300, 300))

  # 100
  grid[69:99, 12:30] = 1
  grid[69:87, 12:90] = 1
  # 108
  grid[69:87, 111:129] = 1
  # 116
  grid[69:99, 165:183] = 1
  grid[69:87, 165:243] = 1
  # 124
  grid[69:87, 264:282] = 1
  ###
  # 102
  grid[138:156, 12:30] = 1
  # 110
  grid[138:156, 51:129] = 1
  grid[126:156, 111:129] = 1
  # 118
  grid[138:156, 165:183] = 1
  # 126
  grid[138:156, 204:282] = 1
  grid[126:156, 268:282] = 1
  ###
  # shift = shift_x(123)
  # 104 = H_100 + shift
  grid[69+123:99+123, 12:30] = 1
  grid[69+123:87+123, 12:90] = 1
  # 106 = H_102 + shift
  grid[138+123:156+123, 12:30] = 1
  # 112 = H_108 + shift
  grid[69+123:87+123, 111:129] = 1
  # 114 = H_110 + shift
  grid[138+123:156+123, 51:129] = 1
  grid[126+123:156+123, 111:129] = 1
  ###
  # 120 = H_116 + shift_x(123)
  grid[69+123:99+123, 165:183] = 1
  grid[69+123:87+123, 165:243] = 1
  # 130 = H_126 + shift_x(123)
  grid[138+123:156+123, 204:282] = 1
  grid[126+123:156+123, 268:282] = 1
  # 128 = H_124 + shift_x(123)
  grid[69+123:87+123, 264:282] = 1
  # bookstore = H_118 + shift_x(123)
  grid[138+123:156+123, 165:183] = 1

  grid = grid.T
  buildings = np.nonzero(grid)
  return grid

def open_chamber(l1, l2, h, k, tau, buildings, GRID_SIZE, steps=int(3e4), eps=1e-5):

  # Evaluate shifting forces
  const_shift = k / h**2
  l1_shift = l1 / (2. * h)
  l2_shift = l2 / (2. * h)

  xshifts = tau * np.array([const_shift - l1_shift, const_shift + l1_shift])
  yshifts = tau * np.array([const_shift - l2_shift, const_shift + l2_shift])

  inplace = 1. - (4 * tau * k) / h**2

  coeffs = np.array([
    inplace,
    xshifts[0], xshifts[1],
    yshifts[0], yshifts[1]
  ])

  NN = GRID_SIZE * GRID_SIZE
  dx = [1, -1, 0, 0]
  dy = [0, 0, 1, -1]

  A = lil_matrix((NN, NN))

  for i in range(GRID_SIZE):
    A[i * GRID_SIZE, i * GRID_SIZE] = 1.0

  # Fulfill matrix with gradient
  for y in range(GRID_SIZE):
    for x in range(1, GRID_SIZE):

      vecidx = y * GRID_SIZE + x
      A[vecidx, vecidx] = coeffs[0]

      for j in range(0, 4):
        x1 = x + dx[j]
        y1 = y + dy[j]
        coeff = coeffs[j + 1]

        if x1 >= GRID_SIZE or y1 >= GRID_SIZE or y1 == 0 or buildings[y1, x1] == 1:
          A[vecidx, vecidx] += coeff
        else:
          A[vecidx, y1 * GRID_SIZE + x1] = coeff

  field = np.zeros((GRID_SIZE, GRID_SIZE))
  field[:, 0] = 1.

  A = csr_matrix(A)
  x = np.reshape(field, (NN, 1))

  # Apply step until converge
  step = 0
  for step in range(steps):
    x_new = A * x
    error = np.linalg.norm(x_new - x) / max(1, np.linalg.norm(x))
    if error < eps:
      break
    x = x_new


  return x.reshape((GRID_SIZE, GRID_SIZE)), step + 1

def draw_2d(m: np.array, metadata):
  plt.imshow(m[:, :], origin='lower', vmin=0.0, vmax=1.0, cmap='hot')
  plt.title(metadata)
  plt.colorbar()
  plt.show()

def output_matrix(matrix, eps):
  np.savetxt('output.txt', matrix)
  with open('output.txt', 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write(f'{eps}\n' + content)

def main():
  N = 300 # grid size
  l1, l2, k = 1., 0., .5
  h = 1.
  tau = .33
  assert tau <= h * h / (4 * k)

  buildings = get_buildings()
  draw_2d(buildings, 'Buildings')
  steps = int(3e4)
  eps = 1e-5

  state, break_step = open_chamber(l1, l2, h, k, tau, buildings, N, steps=steps, eps=eps)
  print(f'Step {break_step}/{steps}, eps={eps}, h={h}, tau={tau}, M={N}')

  # Plot city and store matrix to file
  draw_2d(state, f'Step {break_step}/{steps}, eps={eps}, h={h}, tau={tau}, M={N}')
  output_matrix(state, eps)

if __name__ == '__main__':
  main()
