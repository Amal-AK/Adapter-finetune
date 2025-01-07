from transformers import AutoModel, AutoTokenizer , AutoConfig
from myOpenDelta.opendelta import AdapterModel
from optimization import *
#checkpoint = "Salesforce/codet5-base"

checkpoint = 'Salesforce/codet5p-770m'

set_seed(42)
config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)


print(model.config._name_or_path.lower() )


#for name, module in model.named_modules():
    #print(name)

'''
delta_model = AdapterModel(backbone_model= model ,
                                    modified_modules= ['encoder.block.6.layer.0.SelfAttention' ],
                                    bottleneck_dim=[32] ,
                                    non_linearity= 'relu',
                                    dropout_rate=0.1,
                                    normalization=None,
                                    skip_connection=True
                                    ) 
'''
#x  = random_adapter_parameters(config)
#print(x)
#model = get_delta_model(model , x, 'cuda:0')


delta = AdapterModel(model , bottleneck_dim=[256] , modified_modules= ['encoder.block.1.layer.0.SelfAttention'])
delta.log()
print(model.encoder)


