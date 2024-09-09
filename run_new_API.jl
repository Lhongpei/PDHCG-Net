push!(LOAD_PATH, "src/.")
ENV["CUDA_VISIBLE_DEVICES"] = "2"
import PDHCG
qp = PDHCG.generateProblem("randomqp", n = 100000, density = 1e-4, seed = 0)
initial_primal = ones(size(qp.constraint_matrix, 2))
initial_dual = ones(size(qp.constraint_matrix, 1))
PDHCG.pdhcgSolve(qp, gpu_flag = true, warm_start_flag = false, initial_primal = initial_primal, initial_dual = initial_dual)