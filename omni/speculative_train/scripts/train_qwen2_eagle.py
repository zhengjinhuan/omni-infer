from omni.speculative_train.models.auto import AutoDraftModelConfig, AutoEagleDraftModel

config = AutoDraftModelConfig.from_file("/data/model/qwq-32b-eagle/config.json")
model = AutoEagleDraftModel.from_config(config).npu()
print(model)
names = [item[0] for item in model.named_parameters()]
print(names)

model = AutoEagleDraftModel.from_pretrained("/data/model/qwq-32b-eagle").npu()
print(model)
names = [item[0] for item in model.named_parameters()]
print(names)


