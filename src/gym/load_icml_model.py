import gym
import network_sim
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO1
import tensorflow as tf


def tf_load_model(export_dir):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        graph = tf.get_default_graph()
        model = graph.get_tensor_by_name("finalnode:0")
    return model

env = gym.make('PccNs-v0')
# model = PPO1.load('icml_paper_model')
# env = gym.make('CartPole-v1')
# model = PPO1.load('ppo_cartpole')
# model = tf_load_model('icml_paper_model')
sess = tf.Session()
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
input_key = 'ob'
output_key = 'act'

export_path =  './icml_paper_model'
meta_graph_def = tf.saved_model.loader.load(
           sess,
          [tf.saved_model.tag_constants.SERVING],
          export_path)
signature = meta_graph_def.signature_def

x_tensor_name = signature[signature_key].inputs[input_key].name
y_tensor_name = signature[signature_key].outputs[output_key].name

x = sess.graph.get_tensor_by_name(x_tensor_name)
y = sess.graph.get_tensor_by_name(y_tensor_name)

obs = env.reset()
for i in range(10):
    obs = obs.reshape(1, 30)
    action = sess.run(y, {x: obs})
    obs, rewards, dones, info = env.step(action)
    print(f"rewards: {rewards}")


# obs = env.reset()
# for i in range(10):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     print(rewards)


# default_export_dir = "/tmp/pcc_saved_models/model_A/"
# export_dir = arg_or_default("--model-dir", default=default_export_dir)
# with model.graph.as_default():

#     pol = model.policy_pi#act_model

#     obs_ph = pol.obs_ph
#     act = pol.deterministic_action
#     sampled_act = pol.action

#     obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
#     outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
#     stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(sampled_act)
#     signature = tf.saved_model.signature_def_utils.build_signature_def(
#         inputs={"ob":obs_input},
#         outputs={"act":outputs_tensor_info, "stochastic_act":stochastic_act_tensor_info},
#         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

#     #"""
#     signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                      signature}

#     model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
#     model_builder.add_meta_graph_and_variables(model.sess,
#         tags=[tf.saved_model.tag_constants.SERVING],
#         signature_def_map=signature_map,
#         clear_devices=True)
#     model_builder.save(as_text=True)