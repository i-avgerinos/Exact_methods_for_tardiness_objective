from __future__ import division
import sys
import random
import time
import math
import cplex
from cplex.exceptions import CplexSolverError
from docplex.cp.model import CpoModel
from csv import reader

# -----------------------------------------------------------------------------
# Algorithm 1: LBBD Algorithm
# -----------------------------------------------------------------------------
# Exact methods for tardiness objectives in production scheduling
# -----------------------------------------------------------------------------
# Avgerinos I., Mourtos I., Vatikiotis, S., Zois, G.
# -----------------------------------------------------------------------------
# ELTRUN Research Lab, Department of Management Science and Technology
# Athens University of Economics and Business
# Athens, Greece
# -----------------------------------------------------------------------------

print("---------------------------------------------------------------------------------------------------")
termination_limit = 20 # Maximum number of iterations 'K'
file_name = "Results.txt" # Output file

jobs_vars = [10, 20, 50, 100, 150, 200] # The number of jobs |J| is in this range
machines_vars = [2, 5, 10, 20] # The number of machines |M| is in this range 
workers_vars = [2, 5, 10, 20] # The number of workers R is in this range
dataset_num = 0 # An ID for each generated dataset
for j_fac in range(len(jobs_vars)):
	jobs = jobs_vars[j_fac]
	horizon = int(1440*(jobs/20)) # 'horizon' is the maximum time instance, bigger than any completion time; It will be used as a big number 'V'
	for m_fac in range(len(machines_vars)):
		if machines_vars[m_fac] < jobs_vars[j_fac]: # If |M| >= |J|, no dataset is generated
			machines = machines_vars[m_fac] # The value of |M| is set
			processing_times = [[math.ceil(random.uniform(1, 10)*random.uniform(1, 10) + random.uniform(0, 10)) for m in range(machines)] for i in range(jobs)] #p_{im} computation
			setup_times = [[[math.ceil(random.uniform(0.1, 0.2)*processing_times[j][m])*(i != j) for m in range(machines)] for j in range(jobs)] for i in range(jobs)] #s_{ijm} computation
			zero_setup = [[math.ceil(random.uniform(0.1, 0.2)*processing_times[i][m]) for m in range(machines)] for i in range(jobs)] #'zero_setup[i][m]' stands for s^{0}_{im}

			min_setup = []
			for i in range(jobs):
				min_setup.append([])
				for m in range(machines):
					test_list = []
					for j in range(jobs):
						if i != j:
							test_list.append(setup_times[j][i][m])
					min_setup[i].append(min(test_list)) #min_setup[i][m] stands for \bar{s}_{im}
			for d_fac in range(2): #Two variations of deadlines are considered.
				if d_fac == 0: # 'Loose' deadlines
					tau = 0.8
					rho = 0.2
					P = [min([processing_times[i][m] for m in range(machines)]) + min([min_setup[i][m] for m in range(machines)]) for i in range(jobs)]
				if d_fac == 1: # 'Tight' deadlines
					tau = 0.2
					rho = 0.8
					P = [min([processing_times[i][m] for m in range(machines)]) + min([min_setup[i][m] for m in range(machines)]) for i in range(jobs)] # d_{i} is computed
				deadlines = [int(random.uniform(P[i]*(1 - tau - rho/2), P[i]*(1 - tau + rho/2))) for i in range(jobs)]
				for w_fac in range(len(workers_vars)):
					if workers_vars[w_fac] <= machines_vars[m_fac]: # If R > |M|, no dataset is generated.
						workers = workers_vars[w_fac]
						dataset_num = dataset_num + 1
						with open(file_name, 'a') as output:
							print("Dataset	"+str(dataset_num))
							print("--------------------------------------------")
							if d_fac == 0:
								print("Jobs : "+str(jobs)+",	Machines : "+str(machines)+",	Workers : "+str(workers)+",	Tight")
							if d_fac == 1:
								print("Jobs : "+str(jobs)+",	Machines : "+str(machines)+",	Workers : "+str(workers)+",	Loose")
							start_time = time.time()

							# Master Problem M
							master = cplex.Cplex()
							master.parameters.mip.tolerances.integrality.set(0.0)
							master.set_results_stream(None)
							master.set_warning_stream(None)
							master.parameters.timelimit.set(300) # A time limit of 300 seconds is set for the master problem.

							master_obj = []
							master_lb = []
							master_ub = []
							master_types = []
							master_names = []
							# Variables X_{ijm}
							x = [[["x_"+str(i)+","+str(j)+","+str(m) for m in range(machines)] for j in range(jobs)] for i in range(jobs)]
							for i in range(jobs):
								for j in range(jobs):
									for m in range(machines):
										master_obj.append(0.0)
										master_lb.append(0.0)
										master_ub.append(1.0)
										master_types.append("B")
										master_names.append(x[i][j][m])
							# Variables Y_{im}
							y = [["y_"+str(i)+","+str(m) for m in range(machines)] for i in range(jobs)]
							for i in range(jobs):
								for m in range(machines):
									master_obj.append(0.0)
									master_lb.append(0.0)
									master_ub.append(1.0)
									master_types.append("B")
									master_names.append(y[i][m])
							# Variables C_{jm}
							C = [["C_"+str(j)+","+str(m) for m in range(machines)] for j in range(jobs)]
							for j in range(jobs):
								for m in range(machines):
									master_obj.append(0.0)
									master_lb.append(0.0)
									master_ub.append(cplex.infinity)
									master_types.append("C")
									master_names.append(C[j][m])
							# Variables t^{d}_{jm}
							td = [["td_"+str(i)+","+str(m) for m in range(machines)] for i in range(jobs)]
							for j in range(jobs):
								for m in range(machines):
									master_obj.append(0.0)
									master_lb.append(0.0)
									master_ub.append(cplex.infinity)
									master_types.append("C")
									master_names.append(td[j][m])
							# Variables T_{jm}
							T = [["T_"+str(i)+","+str(m) for m in range(machines)] for i in range(jobs)]
							for j in range(jobs):
								for m in range(machines):
									master_obj.append(0.0)
									master_lb.append(0.0)
									master_ub.append(cplex.infinity)
									master_types.append("C")
									master_names.append(T[j][m])
							# Objective value 'z'
							z = ["z"]
							master_obj.append(1.0)
							master_lb.append(0.0)
							master_ub.append(cplex.infinity)
							master_types.append("C")
							master_names.append(z[0])

							master.variables.add(obj = master_obj,
												 lb = master_lb,
												 ub = master_ub,
												 types = master_types,
												 names = master_names)

							master_expressions = []
							master_senses = []
							master_rhs = []

							for i in range(jobs):
								# Constraints (2)
								constraint = cplex.SparsePair(ind = [y[i][m] for m in range(machines)],
															  val = [1.0 for m in range(machines)])
								master_expressions.append(constraint)
								master_senses.append("E")
								master_rhs.append(1.0)
								for m in range(machines):
									# Constraints (3)
									constraint = cplex.SparsePair(ind = [y[i][m]] + [x[i][j][m] for j in range(jobs)],
														 		  val = [1.0]     + [-1.0 for j in range(jobs)])
									master_expressions.append(constraint)
									master_senses.append("E")
									master_rhs.append(0.0)
							for m in range(machines):
								for j in range(jobs):
									# Constraints (4)
									constraint = cplex.SparsePair(ind = [x[i][j][m] for i in range(jobs)],
																  val = [1.0 for i in range(jobs)])
									master_expressions.append(constraint)
									master_senses.append("L")
									master_rhs.append(1.0)
								for j in range(jobs):
									if j == 0:
										# Constraints (5)
										constraint = cplex.SparsePair(ind = [C[j][m]] + [x[i][j][m] for i in range(jobs)],
																	  val = [1.0]     + [-processing_times[i][m]-min_setup[i][m] for i in range(jobs)])
										master_expressions.append(constraint)
										master_senses.append("E")
										master_rhs.append(0.0)
									if j > 0:
										# Constraints (6)
										constraint = cplex.SparsePair(ind = [C[j][m]] + [C[j-1][m]] + [x[i][j][m] for i in range(jobs)],
																	  val = [1.0]     + [-1.0]      + [-processing_times[i][m]-min_setup[i][m] for i in range(jobs)])
										master_expressions.append(constraint)
										master_senses.append("E")
										master_rhs.append(0.0)
									# Constraints (7)
									constraint = cplex.SparsePair(ind = [td[j][m]] + [x[i][j][m] for i in range(jobs)],
																  val = [1.0]      + [-deadlines[i] for i in range(jobs)])
									master_expressions.append(constraint)
									master_senses.append("E")
									master_rhs.append(0.0)
									# Constraints (8)
									constraint = cplex.SparsePair(ind = [T[j][m]] + [td[j][m]] + [C[j][m]],
																  val = [1.0]     + [1.0]      + [-1.0])
									master_expressions.append(constraint)
									master_senses.append("G")
									master_rhs.append(0.0)
							# Constraints (1); Objective Function
							objective_function = cplex.SparsePair(ind = [z[0]] + [T[i][m] for i in range(jobs) for m in range(machines)],
																  val = [1.0]  + [-1.0 for i in range(jobs) for m in range(machines)])
							master_expressions.append(objective_function)
							master_senses.append("G")
							master_rhs.append(0.0)

							master.linear_constraints.add(lin_expr = master_expressions, 
														  senses = master_senses,
														  rhs = master_rhs)

							master.objective.set_sense(master.objective.sense.minimize)
							master.solve()

							best_lb = 0 # The best Lower Bound is initialized.
							best_ub = horizon*100 # The best Upper Bound is initialized.
							best_gap = 1000 # The best Gap is initialized.

							print("------------------------------------------------------------------------------------")
							print("#		LB		UB		Gap		Best LB		Best UB		Best Gap")
							print("------------------------------------------------------------------------------------")

							for iteration in range(termination_limit): # 'iteration' stands for 'k'; Must be less than K
								lower_bound = round(master.solution.MIP.get_best_objective(), 2) # The Lower Bound of iteration 'k' is set.
								if lower_bound > best_lb: 
									best_lb = lower_bound # If the incumbent best lower bound is worse than the lower bound of iteration 'k', the value is updated.
								assignments = [[] for m in range(machines)] # Set \hat{M} is defined.
								for j in range(jobs):
									for m in range(machines):
										for i in range(jobs):
											if master.solution.get_values(x[i][j][m]) > 0.9:
												assignments[m].append(i)

								# Subproblem S
								subproblem = CpoModel()
								# Interval Variables \sigma_{mj}
								setup = {}
								for m in range(machines):
									for i in range(len(assignments[m])):
										if i == 0:
											start = (0, horizon)
											end = (0, horizon)
											size = math.ceil(zero_setup[assignments[m][i]][m]) # Constraints (15)
											setup[(m, i)] = subproblem.interval_var(start, end, size, name = "setup"+str(m)+","+str(assignments[m][i]))
										else:
											start = (0, horizon)
											end = (0, horizon)
											size = int(setup_times[assignments[m][i-1]][assignments[m][i]][m]) # Constraints (14)
											setup[(m, i)] = subproblem.interval_var(start, end, size, name = "setup"+str(m)+","+str(assignments[m][i]))
								# Variables C*_{mj}
								completion_times = []
								for m in range(machines):
									completion_times.append([])
									for i in range(len(assignments[m])):
										completion_times[m].append(subproblem.integer_var(0, horizon, "Ct_"+str(m)+","+str(i)))
								# Variables T*_{i}
								tardiness = []
								for i in range(jobs):
									tardiness.append(subproblem.integer_var(0, horizon - deadlines[i], "T_"+str(i)))
								# Constraints (10)
								subproblem.add(sum([subproblem.pulse(setup[(m, i)], 1) for m in range(machines) for i in range(len(assignments[m]))]) <= workers)
								for m in range(machines):
									for i in range(len(assignments[m])):
										if i == 0:
											# Constraints (13) (for the first slot of machine m)
											subproblem.add(completion_times[m][i] == setup_times[0][assignments[m][i]][m] + processing_times[assignments[m][i]][m])
										else:
											# Constraints (13)
											subproblem.add(subproblem.start_of(setup[(m, i)]) >= subproblem.end_of(setup[(m, i-1)]) + processing_times[assignments[m][i-1]][m])
											# Constraints (11)
											subproblem.add(completion_times[m][i] == subproblem.end_of(setup[(m, i)]) + processing_times[assignments[m][i]][m])
										# Constraints (12)
										subproblem.add(tardiness[assignments[m][i]] >= completion_times[m][i] - deadlines[assignments[m][i]])
								# Constraints (9); Objective Function
								total_cost = sum([tardiness[i] for i in range(jobs)]) 
								subproblem.add(subproblem.minimize(total_cost))

								sol = subproblem.solve(TimeLimit = 60, trace_log = False) # A time limit of 60 seconds is arbitrarily set; it is never reached.

								upper_bound = 0
								for i in range(jobs):
									upper_bound = upper_bound + sol[tardiness[i]] # The upper bound of iteration 'k' is set.
								if upper_bound < best_ub: # If a better upper bound is found, the value of the best upper bound is updated.
									best_ub = upper_bound
								if upper_bound > 0:
									gap = round(100*(upper_bound - lower_bound)/upper_bound, 2) # The value of Gap is computed.
								if upper_bound == 0:
									gap = round(0.0, 2)
								if best_ub > 0:
									best_gap = round(100*(best_ub - best_lb)/best_ub, 2)
								if best_ub == 0:
									best_gap = round(0.0, 2)
								print(str(iteration)+"		"+str(round(lower_bound,2))+"	"+str(round(upper_bound,2))+"		"+str(gap)+"%"+"	"+str(round(best_lb, 2))+"		"+str(round(best_ub, 2))+"			"+str(best_gap)+"%")

								if gap > 0.1: # The maximum gap of termination is set to E = 0.1%. If the Gap of iteration 'k' is greater than E, a set of otpimality cuts is added to M.
									previous_x = [] # 'previous_x' is the set of assignments A_{k-1}
									for i in range(jobs):
										for j in range(jobs):
											for m in range(machines):
												if master.solution.get_values(x[i][j][m]) > 0.9:
													previous_x.append([i, j, m]) # For each assignment a, a0 = i, a1 = j, a2 = m for which X_{ijm} = 1 in the previous iteration.
									# zeta will replace 'z' in the objective function of M
									zeta = ["zeta"]
									master.variables.add(obj = [0.0],
														 lb = [0.0],
														 ub = [cplex.infinity],
														 types = ["C"],
														 names = [zeta[0]])
									# Constraints (16)
									cut = cplex.SparsePair(ind = [zeta[0]] + [x[previous_x[a][0]][previous_x[a][1]][previous_x[a][2]] for a in range(len(previous_x))],
														   val = [1.0]     + [-upper_bound for a in range(len(previous_x))])
									master.linear_constraints.add(lin_expr = [cut], 
																  senses = ["G"],
																  rhs = [upper_bound - upper_bound*len(previous_x)])
									# The objective function is updated.
									cut = cplex.SparsePair(ind = [z[0]] + [zeta[0]],
														   val = [1.0]  + [-1.0])
									master.linear_constraints.add(lin_expr = [cut], 
																  senses = ["G"],
																  rhs = [0.0])

									master.objective.set_sense(master.objective.sense.minimize)
									master.solve()
								if gap <= 0.1: # If the Gap of iteration 'k' reaches the limit E, the algorithm terminates.
									break
							end_time = time.time()
							output.write(str(dataset_num)+";"+str(jobs)+";"+str(machines)+";"+str(workers)+";"+str(d_fac)+";"+str(best_lb)+";"+str(best_ub)+";"+str(int(end_time - start_time))+";"+str(iteration)+"\n") # The results of Algorithm 1 for this instance is written in the output file.
							print("------------------------------------------------------------------")
						output.close()