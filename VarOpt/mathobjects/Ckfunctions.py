import casadi as cs
import os
from IPython import embed
class Ckfunction(cs.Function):
	"""generalization of cs.Function to Ck class functions defined on a manifold. This class of functions also forms a vector space on the real and complex fields"""

	def __init__(self, name, inargs, outargs, k=1):
		if not (outargs[0].__class__==Ckfunction):
			super(Ckfunction, self).__init__(name, inargs, outargs)
			self.output_type = 'symbolic'
		else: # if outarg itself is a Ckfunction
			out = [];
			# self.output_arg = []
			for k in range(len(outargs)):
				f = outargs[k];
				out.append(f(*f.var_in()))
			self.output_arg=f.var_in() # currently not supporting multiple function outputs with mutiple input argument interfaces
			super(Ckfunction, self).__init__(name, inargs, out)
			self.output_type = 'Ckfunction'
		self.k = k

		# self.Manifold = Manifold(dim=1,symtype=cs.SX)

	def __call__(self,*inargs):
		f0 = super(Ckfunction,self).__call__(*inargs)
		if (self.output_type=='Ckfunction') and (not f0.is_constant()):
			# embed()
			f = Ckfunction(self.name(),self.output_arg,[f0])
		else:
			f = f0;
		# embed()
		return f#Ckfunction(self.name()+'_arg',)

	def var_in(self):
		if self.type_name() == 'sxfunction':
			return self.sx_in()
		else:
			return self.mx_in()

	def var_free(self):
		if self.type_name() == 'sxfunction':
			return self.free_sx()
		else:
			return self.free_mx()

	def __compose__(self, other):
		# embed()
		return Ckfunction(self.name() + '_o_' + other.name(), other.var_in(), [self.__call__(other(*other.var_in()))])

	def Add(self,other):
		if not (other.__class__==Ckfunction):
			return Ckfunction('c_p_'+self.name(),self.var_in(),[other+self.__call__(*self.var_in())])
		else:
			# embed()
			return Ckfunction(self.name() + '_p_' + other.name(), other.var_in(), [self.__call__(*other.var_in()) + other(*other.var_in())])

	def Sub(self,other):
		if not (other.__class__==Ckfunction):
			return Ckfunction(self.name()+'_m_c',self.var_in(),[self.__call__(*self.var_in())-other])
		else:
			# embed()
			return Ckfunction(self.name() + '_m_' + other.name(), other.var_in(), [self.__call__(*other.var_in()) - other(*other.var_in())])

	def Mul(self,other):
		# if (other.__class__==cs.SX)or(other.__class__==cs.MX):
		if not other.__class__==Ckfunction:
			return Ckfunction('c_'+self.name(),self.var_in(),[other*self.__call__(*self.var_in())])
		else:
			return Ckfunction('c_'+self.name(),self.var_in(),[other.__call__(*self.var_in())*self.__call__(*self.var_in())])

	def __rmul__(self,other):
		return self.Mul(other)

	def __mul__(self,other):
		return self.Mul(other)

	def __add__(self, other):
		"""assumes common arguments and adds two functions to give another function"""
		return self.Add(other)

	def __radd__(self, other):
		"""assumes common arguments and adds two functions to give another function"""
		return self.Add(other)

	def __iadd__(self,other):
		return self.Add(other)

	def __sub__(self,other):
		return self.Sub(other)

	def __rsub__(self,other):
		return self.Sub(other)

	def generate(self):
		# generate the scalar function
		name = self.name()
		opts = {'with_header':True}
		super(Ckfunction,self).generate(name+'.c',opts)
		os.system("mkdir -p src/"+name)
		os.system("mv "+name+'.c '+"src/"+name)
		os.system("mv "+name+'.h '+"src/"+name)

	def compile(self):
		name = self.name()
		os.system("gcc -fPIC -shared src/"+name+"/"+name+".c -o src/"+name+"/"+name+".so")

	def OpenClGen(self):
		name = self.name()
		os.system("mkdir -p src/"+name)
		code = '#include \"'+name+'.h\"\n';
		code+='__kernel void '+name+'_kernel'+' (__global double** arg, __global double** res)\n'
		code+='{\n'
		code+='const int i = get_global_id (0);\n'
		code+=name+'(arg[i],res[i],0,0,0);\n'
		code+='}'
		with open("src/"+name+'/'+name+'.cl','w') as f:
			f.write(code)
