
#-------------Running Parameters--------------
#-----------------GPU setting-----------------
GPU_on = 1                     # 1: use GPU; 0: use CPU
GPU_id = 1                     # The GPU id if there are multiple GPUs (if there is only one GPU, set it to 0)
#-----------------Task setting----------------  
time_limit = 3600              # The time limit for each problem
relat = 1e-6                   # The relative tolerance for the solver
save_path = "saved_results/QP/$GPU_on"
scale = 1000              # The scale of the problem
#---------------------------------------------
#--------------------END----------------------


function run_solver(generator, save_path, use_gpu=0, GPU_id=0, time_limit=3600, relat=1e-6, scale=100000)
    "
     `file_path`: Path to the quadratic programming instance file.
     `save_path`: Directory where the output files will be saved.
     `use_gpu`: Enables GPU acceleration if set to 1; otherwise, it remains on CPU (default: 0).
     `GPU_id`: Identifies which GPU to use if GPU acceleration is enabled (default: 0).
     `time_limit`: Sets the maximum allowed time for the solver to run in seconds (default: 3600).
     `relat`: Specifies the solver's relative tolerance level (default: 1e-6).
     "
    project_scr = ["--project=scripts", "./scripts/solve_generated.jl"]
    time_limit_arg = ["--time_sec_limit", "$time_limit"]
    relat_arg = ["--tolerance", "$relat"]
    gpu_option = use_gpu == 1 ? ["--use_gpu", "1"] : ["--use_gpu", "0"]
    out_dir = ["--output_dir", "$save_path"]
    gener = ["--generator", generator]
    scale = ["--scale", "$scale"]
    randseed = ["--randseed", "1"]
    if use_gpu == 1
        ENV["CUDA_VISIBLE_DEVICES"] = "$GPU_id"
    end

    local_problem = `julia $project_scr $gener $out_dir $relat_arg $time_limit_arg $gpu_option $scale $randseed`
    println("Running command: ", local_problem)
    try
        run(local_problem)
        return true
    catch e
        println("Failed to solve, due to error: $e")
        return false
    end
end

run_solver("mpc", save_path, GPU_on, GPU_id, time_limit, relat, scale)