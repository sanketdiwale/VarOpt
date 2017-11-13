import casadi as cs
# from MyMathApp.mathobjects.VectorSpace import VectorSpace 
from MyMathApp.mathobjects.Ckfunctions import Ckfunction
from MyMathApp.mathobjects.ExpressionManager import Expressions
from IPython import embed
class Kernel(object):
	"""Kernel:  maps X x X --> L(Y)"""
	def __init__(self,indexshape=(1,1),outshape=(1,1),symtype=cs.SX):
		super(Kernel, self).__init__()
		self.indexshape = indexshape  # shape of representer for element in set X
		self.outshape = outshape
		self.symtype = symtype
		self.__params__ = self.symtype.sym('kern_par',0,0);
		self.__paramsval__ = cs.DM.ones(self.__params__.shape)
		self.xsym = self.symtype.sym('ker_x',self.indexshape);
		self.ysym = self.symtype.sym('ker_y',self.indexshape);
		self.name = 'kernel'
		self.kernel = Ckfunction(self.name,[self.xsym,self.ysym,self.__params__],[1.]) # dummy kernel
		self.Expressions = Expressions()

	def __call__(self,x1,x2,params=None):
		if params is None:
			params = self.__paramsval__
		if not ((x1.shape[1]==1) and (x2.shape[1]==1)): # Broadcast the inputs
			f1 = self.kernel.map('rowexp','serial',x2.shape[1],[0,2],[])
			f2 = f1.map('colexp','serial',x1.shape[1],[1,2],[])
			try:
				k1 = cs.blocksplit(f2(x1,x2,params),self.outshape[0],self.outshape[0]*x2.shape[1])[0];
			except:
				embed()
			return cs.vertcat(*k1).T
		else: # Scalar evaluation
			return self.kernel(x1,x2,params)
	
	def __Add__(self,other):
		K = Kernel(indexshape=self.indexshape,outshape=self.outshape,symtype=self.symtype)
		K.kernel = self.kernel + other.kernel
		K.xsym = self.xsym
		K.ysym = self.ysym
		K.__params__ = cs.vertcat(self.__params__,other.__params__)
		K.__paramsval__ = cs.vertcat(self.__paramsval__,other.__paramsval__)
		return K

	def __add__(self,other):
		return self.__Add__(other)

	def __radd__(self,other):
		return self.__Add__(other)

	def __Mul__(self,other):
		K = Kernel(indexshape=self.indexshape,outshape=self.outshape,symtype=self.symtype)
		if other.__class__==Kernel:
			K.kernel = self.kernel * other.kernel
			K.__params__ = cs.vertcat(self.__params__,other.__params__)
			K.__paramsval__ = cs.vertcat(self.__paramsval__,other.__paramsval__)
		else:
			K.kernel = self.kernel * other
			K.__params__ = self.__params__
			K.__paramsval__ = self.__paramsval__
		K.xsym = self.xsym
		K.ysym = self.ysym
		return K

	def __rmul__(self,other):
		return self.__Mul__(other)

	def __mul__(self,other):
		return self.__Mul__(other)

	def __DirectSum__(self,other,name='kernel'):
		K = Kernel(indexshape=self.indexshape,outshape=(self.outshape[0]+other.outshape[0],self.outshape[1]+other.outshape[1]),symtype=self.symtype)
		K.xsym = self.xsym
		K.ysym = self.ysym
		for name in self.Expressions.collec['Params']['names']:	
			K.Expressions.add(member='Params',expr=self.Expressions.getexpr('Params',name),eq=self.Expressions.getparam(name),name=name+'0')
		for name in other.Expressions.collec['Params']['names']:	
			K.Expressions.add(member='Params',expr=other.Expressions.getexpr('Params',name),eq=other.Expressions.getparam(name),name=name+'1')
		K.__params__ = K.Expressions.flatten('Params','expr')
		K.__paramsval__ = K.Expressions.flatten('Params','val')
		# embed()
		K.kernel = Ckfunction(name,[self.xsym,self.ysym,K.__params__],[cs.diag(cs.vertcat(self.__call__(self.xsym,self.ysym,self.__params__),other.__call__(self.xsym,self.ysym,other.__params__)))])
		return K

	def generate(self):
		self.kernel.generate()
		self.kernel.OpenClGen()
# class MultioutKernel(Kernel):
# 	"""docstring for MultioutKernel"""
# 	def __init__(self,dim = 1,indexshape=(1,1),outshape=(1,1),symtype=cs.SX):
# 		super(MultioutKernel, self).__init__(dim,indexshape,symtype)
# 		self.outsym = symtype.sym('Y',outshape)
# 		self.kernel = Ckfunction('kernel',[self.xsym,self.ysym,self.__params__],[Ckfunction('op',[self.outsym],[self.outsym])]) # dummy kernel