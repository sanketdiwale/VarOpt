import casadi as cs
import numpy as np
from MyMathApp.mathobjects.ExpressionManager import Expressions
from multiprocessing.pool import ThreadPool
from time import sleep
from IPython import embed
class VarProb(object):
	"""docstring for VarProb"""
	def __init__(self, name):
		super(VarProb, self).__init__()
		self.name = name
		self.Expressions = Expressions()
		self.solver_ready=False
		self.solver_pool = ThreadPool(processes=1);
		self.sleeptime = 0.05;
		self.QP = False
		# self.solve_stats = {
		# 'avgsoltime':ringbuffer((1000,1)),'varsoltime':0,'failrate':freqcount(nbins=2,nhistory=100),
		# 'avgcallfreq':0}

	def problem_setup(self):
		pass

	def GENCODE(self):
		self.problem_setup()
		self.nlp = {'f':self.Expressions.flatten('Objective','expr'),'x':self.Expressions.flatten('DesVar','expr'),'g':self.Expressions.flatten('Constraints','expr'),'p':self.Expressions.flatten('Params','expr')}
		if not self.QP:
			opts = {}
			opts["ipopt.warm_start_init_point"] = "yes";
			# opts["ipopt.hessian_approximation"] = "limited-memory"
			self.optimizer = cs.nlpsol('solver','ipopt',self.nlp,opts);
		else:
			opts = {'sparse':True}
			self.optimizer = cs.qpsol('solver','qpoases',self.nlp,opts);
		self.args = {}
		# embed()
		self.args["lbx"] = self.Expressions.flatten('DesVar','lb')
		self.args["ubx"] = self.Expressions.flatten('DesVar','ub')
		self.args["lbg"] = self.Expressions.flatten('Constraints','lb')
		self.args["ubg"] = self.Expressions.flatten('Constraints','ub')
		self.args["p"]   = self.Expressions.flatten('Params','val')
		self.solver_ready=True

	def paramupdate(self):
		self.args["p"]   = self.Expressions.flatten('Params','val')
		
	def call(self):
		if self.solver_ready:
			self.solver_ready=False
			# embed()
			self.optim_thread = self.solver_pool.apply_async(self.optimizer,args=(),kwds=self.args) # call solver in a separate thread

	def getsol(self,skip=False):
		if skip:
			if self.optim_thread.ready(): # if solution is ready extract
				self.sol = self.optim_thread.get()
				self.solver_ready = True; # set true to enable next launch of solver
				return self.sol
		else:
			while not self.optim_thread.ready():
				sleep(self.sleeptime)
				self.sol = self.optim_thread.get()
				self.solver_ready = True; # set true to enable next launch of solver
				return self.sol
		# if not self.QP:
		# 	if stats['return_status']=='Solve_Succeeded':
		# 		self.solve_stats['failrate'].update(0)
		# 	else:
		# 		self.solve_stats['failrate'].update(1)
		# 	self.solve_stats['avgsoltime'].update(stats['t_proc_mainloop'])
		# else:
		# 	pass # current the qpsol returns empty dictionary for stats

	def closesolverpool(self):
		self.solver_pool.close()
		self.solver_pool.join()
		