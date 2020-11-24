from core.human_box import MoCap

mc = MoCap("143")
trajectories, contexts = mc.get_demonstrations(20)