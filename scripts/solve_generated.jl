push!(LOAD_PATH, "src/.")
import ArgParse
import GZip
import JSON3

import CUDA

import PDHCG
using LinearAlgebra

function write_vector_to_file(filename, vector)
    open(filename, "w") do io
      for x in vector
        println(io, x)
      end
    end
end

function solve_instance_and_output(
    qp::PDHCG.QuadraticProgrammingProblem,
    parameters::PDHCG.PdhcgParameters,
    output_dir::String,
    gpu_flag::Bool,
    saved_name::String,
)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
  
    instance_name = saved_name
  
    function inner_solve()
        if gpu_flag
            output = PDHCG.optimize_gpu(parameters, qp)
        else
            output = PDHCG.optimize(parameters, qp)
        end

        log = PDHCG.SolveLog()
        log.instance_name = instance_name
        log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
        log.termination_reason = output.termination_reason
        log.termination_string = output.termination_string
        log.iteration_count = output.iteration_count
        log.CG_total_iteration = output.CG_total_iteration
        log.solve_time_sec = output.iteration_stats[end].cumulative_time_sec
        log.solution_stats = output.iteration_stats[end]
        kkt_error =  Vector{Float64}()
        for i = 1:length(output.iteration_stats)
            c_i_current = output.iteration_stats[i].convergence_information[1]
            current_kkt_err = norm([c_i_current.relative_optimality_gap, c_i_current.relative_l2_primal_residual, c_i_current.relative_l2_dual_residual])
            
            push!(kkt_error,current_kkt_err)
        end
        log.kkt_error = kkt_error

        log.solution_type = PDHCG.POINT_TYPE_AVERAGE_ITERATE
    
        summary_output_path = joinpath(output_dir, instance_name * "_summary.json")
        open(summary_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
    
        log.iteration_stats = output.iteration_stats
        full_log_output_path =
            joinpath(output_dir, instance_name * "_full_log.json.gz")
        GZip.open(full_log_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
    
        primal_output_path = joinpath(output_dir, instance_name * "_primal.txt")
        write_vector_to_file(primal_output_path, output.primal_solution)
    
        dual_output_path = joinpath(output_dir, instance_name * "_dual.txt")
        write_vector_to_file(dual_output_path, output.dual_solution)
    end     

    inner_solve()
   
    return
end

function warm_up(qp::PDHCG.QuadraticProgrammingProblem, gpu_flag::Bool,)
    restart_params = PDHCG.construct_restart_parameters(
        PDHCG.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        PDHCG.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.2,                    # primal_weight_update_smoothing
    )

    termination_params_warmup = PDHCG.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = 1.0e-3,
        eps_optimal_relative = 1.0e-3,
        time_sec_limit = Inf,
        iteration_limit = 10,
        kkt_matrix_pass_limit = Inf,
    )

    params_warmup = PDHCG.PdhcgParameters(
        10,
        true,
        1.0,
        1.0,
        true,
        0,
        true,
        40,
        termination_params_warmup,
        restart_params,
        PDHCG.ConstantStepsizeParams(),
    )
    if gpu_flag
        PDHCG.optimize_gpu(params_warmup, qp);
    else
        PDHCG.optimize(params_warmup, qp);
    end
end


function parse_command_line()
    arg_parse = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table! arg_parse begin
        "--generator"
        help = "The generator of the problem with choices: [randomqp, lasso, svm, portfolio, mpc]."
        arg_type = String
        default = "randomqp"

        "--output_directory"
        help = "The directory for output files."
        arg_type = String
        required = true

        "--tolerance"
        help = "KKT tolerance of the solution."
        arg_type = Float64
        default = 1e-3

        "--time_sec_limit"
        help = "Time limit."
        arg_type = Float64
        default = 3600.0

        "--use_gpu"
        help = "Using GPU: 0-false, 1-true"
        arg_type = Int64
        default = 0

        "--randseed"
        help = "Random seed used to generate problem."
        arg_type = Int64
        default = 42

        "--scale"
        help = "Scale of the problem."
        arg_type = Int64
        default = 100000

    end

    return ArgParse.parse_args(arg_parse)
end


function main()
    parsed_args = parse_command_line()
    generator = parsed_args["generator"]
    tolerance = parsed_args["tolerance"]
    time_sec_limit = parsed_args["time_sec_limit"]
    output_directory = parsed_args["output_directory"]
    gpu_flag = Bool(parsed_args["use_gpu"])
    randseed = parsed_args["randseed"]
    scale = parsed_args["scale"]
    
    if gpu_flag && !CUDA.functional()
        error("CUDA not found when --use_gpu=1")
    end
    if generator == "randomqp"
        qp = PDHCG.generate_randomQP_problem(scale, randseed)
        saved_name = "randomqp_seed$(randseed)_scale$(scale)"
    elseif generator == "lasso"
        qp = PDHCG.generate_lasso_problem(scale, randseed)
        saved_name = "lasso_seed$(randseed)_scale$(scale)"
    elseif generator == "svm"
        qp = PDHCG.generate_svm_problem(scale, randseed)
        saved_name = "svm_seed$(randseed)_scale$(scale)"
    elseif generator == "portfolio"
        qp = PDHCG.generate_portfolio_problem(scale, randseed)
        saved_name = "portfolio$(randseed)_scale$(scale)"
    elseif generator == "mpc"
        qp = PDHCG.generate_mpc_problem(scale, randseed)
        saved_name = "mpc_seed$(randseed)_scale$(scale)"
    else
        error("Unknown generator: ", generator)
    end
    println("Generated Successfully")
    qpw = copy(qp)
    oldstd = stdout
    redirect_stdout(devnull)
    warm_up(qpw, gpu_flag);
    redirect_stdout(oldstd)

    restart_params = PDHCG.construct_restart_parameters(
        PDHCG.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        PDHCG.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.2,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.2,                    # primal_weight_update_smoothing
    )

    termination_params = PDHCG.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        time_sec_limit = time_sec_limit,
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )

    params = PDHCG.PdhcgParameters(
        10,
        false,
        1.0,
        1.0,
        false,
        2,
        true,
        40,
        termination_params,
        restart_params,
        PDHCG.ConstantStepsizeParams(),  
    )

    solve_instance_and_output(
        qp,
        params,
        output_directory,
        gpu_flag,
        saved_name,
    )

end

main()
