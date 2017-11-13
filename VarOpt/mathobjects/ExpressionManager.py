import casadi as cs
import numpy as np
class Expressions(object):
	"""docstring for Constraints"""
	def __init__(self):
		super(Expressions, self).__init__()
		self.collec = {'Constraints':{'expr':[],'lb':[],'ub':[],'names':[]},'Objective':{'expr':[0],'names':['total']},'Functions':{'expr':[],'names':[]},'DesVar':{'expr':[],'lb':[],'ub':[],'warm_vals':[],'names':[]},'Params':{'expr':[],'val':[],'names':[]}}

	def add(self,member,expr,name,eq=None,lb=None,ub=None,withslack=False):
		if ((member=='Constraints') or (member=='DesVar')):
			if ((member=='Constraints') and(withslack)):
				if eq is not None:
					lb = eq; ub = eq;
				eps = cs.SX.sym('slack'+name,expr.shape)
				self.add('DesVar',eps,'slack'+name,lb=np.zeros(expr.shape),ub=np.inf*np.ones(expr.shape))
				self.add('Constraints',expr+eps,name+'lb',lb=lb,ub=np.inf*np.ones(expr.shape))
				self.add('Constraints',expr-eps,name+'ub',lb=-np.inf*np.ones(expr.shape),ub=ub)
				self.collec['Objective']['expr'][0] += 1e8*cs.sum2(cs.sum1(eps));
			else:
				if (eq is not None) and(lb is None) and(ub is None):
					self.collec[member]['expr'].append(expr)
					self.collec[member]['lb'].append(eq)
					self.collec[member]['ub'].append(eq)
				elif (eq is None) and(lb is not None) and(ub is not None):
					self.collec[member]['expr'].append(expr)
					self.collec[member]['lb'].append(lb)
					self.collec[member]['ub'].append(ub)
				else:
					raise ValueError('Incorrect arguments, '+member+' not added')
			if member=='DesVar':
				self.collec[member]['warm_vals'].append(np.zeros(expr.shape))
		elif member=='Objective':
			self.collec[member]['expr'][0] += expr
		elif member=='Functions':
			self.collec[member]['expr'].append(expr)
		elif member=='Params':
			self.collec[member]['expr'].append(expr)
			self.collec[member]['val'].append(eq)
		
		self.collec[member]['names'].append(name)
		
	def setup_eval(self,sol):
		for key,val in self.collec.iteritems():
			pass

	def generateC(self,name):
		import os
		os.system("mkdir -p "+name+"/src")
		for member in ['Constraints','Objective']:
			F = cs.Function(member,[self.flatten('DesVar','expr'),self.flatten('Params','expr')],[self.flatten(member,'expr')])
			F.generate(member+'.c')
			os.system("mv "+member+".c "+name+"/src/")
			jf = cs.jacobian(self.flatten(member,'expr'),self.flatten('DesVar','expr'));
			JF = cs.Function(member+'_jac',[self.flatten('DesVar','expr'),self.flatten('Params','expr')],[jf])
			JF.generate(member+'_jac.c')
			os.system("mv "+member+"_jac.c "+name+"/src/")
			jjf= cs.jacobian(jf.T,self.flatten('DesVar','expr'))
			JJF= cs.Function(member+'_hess',[self.flatten('DesVar','expr'),self.flatten('Params','expr')],[jjf])
			JF.generate(member+'_hess.c')
			os.system("mv "+member+"_hess.c "+name+"/src/")	
		
	def flatten(self,member,key):
		return cs.vertcat(*[cs.reshape(x,-1,1) for x in self.collec[member][key]])

	def get(self,val,member,name):
		f = cs.Function('f',[self.flatten('DesVar','expr'),self.flatten('Params','expr')],[self.collec[member]['expr'][self.collec[member]['names'].index(name)]])
		return f(val,self.flatten('Params','val'))

	def setparam(self,name,val):
		self.collec['Params']['val'][self.collec['Params']['names'].index(name)] = val;

	def getparam(self,name):
		return self.collec['Params']['val'][self.collec['Params']['names'].index(name)];

	def getexpr(self,member,name):
		return self.collec[member]['expr'][self.collec[member]['names'].index(name)]