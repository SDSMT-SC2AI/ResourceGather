import argparse, json, os, errno


def parse_args():
    parser = argparse.ArgumentParser(description="StarCraft II agent A2C TensorFlow implementation")
    parser.add_argument("--max_mineral_cost",
                        default=300, type=int,
                        help="Maximum mineral cost of a unit")

    parser.add_argument("--max_gas_cost",
                        default=50, type=int,
                        help="Maximum gas cost of a unit")

    parser.add_argument("--max_bases",
                        default=5, type=int,
                        help="Maximum number of bases allowed for the agent")

    parser.add_argument("--config",
                        default="config.json", type=str,
                        help="Configuration file")

    parser.add_argument("--train",
                        default=True, type=bool,
                        help="To train or not to train")

    parser.add_argument("--render",
                        default=True, type=bool,
                        help="Render feature layers with pygame")

    parser.add_argument("--episodes",
                        default=200, type=int,
                        help="Total number of episodes to run agent with")

    parser.add_argument("--screen_resolution",
                        default=84, type=int,
                        help="Resolution for screen feature layers")

    parser.add_argument("--minimap_resolution",
                        default=64, type=int,
                        help="Resolution for minimap feature layers")

    parser.add_argument("--game_steps_per_episode",
                        default=0, type=int,
                        help="Game steps per episode")

    parser.add_argument("--step_mul",
                        default=1, type=int,
                        help="Game steps per agent step")

    parser.add_argument("--agent",
                        default="Dummy", type=str,
                        help="ClassName of agent")

    parser.add_argument("--agent_module",
                        default="agent", type=str,
                        help="Path to agent module")

    parser.add_argument("--agent_race",
                        default="Z", type=str,
                        help="Race of agent")

    parser.add_argument("--bot_race",
                        default="R", type=str,
                        help="Race of opponent bot")

    parser.add_argument("--difficulty",
                        default=None,
                        help="Bot's strength")

    parser.add_argument("--profile",
                        default=False, type=bool,
                        help="Whether to turn on code profiling")

    parser.add_argument("--trace",
                        default=False, type=bool,
                        help="Whether to trace the code execution")

    parser.add_argument("--num_envs",
                        default=4, type=int,
                        help="How many instances to run in parallel")

    parser.add_argument("--save_replay",
                        default=True, type=bool,
                        help="Whether to save a replay on completion")

    parser.add_argument("--map",
                        default="AbyssalReefLE_RL", type=str,
                        help="Unused! The AGENT should define the map/maps!")

    parser.add_argument("--base_output_dir",
                        default=".out", type=str,
                        help="Place to put program output")

    parser.add_argument("--summary_dir",
                        default="summary", type=str,
                        help="Place to put summary relative to base output dir")

    parser.add_argument("--checkpoint_dir",
                        default="checkpoints", type=str,
                        help="Place to put model checkpoints relative to base output dir")

    parser.add_argument("--logging_dir",
                        default="out", type=str,
                        help="Place to put logging output relative to base output dir")

    parser.add_argument("--test_dir",
                        default="test", type=str,
                        help="Place for test output relative to base output dir")

    args = parser.parse_args()

    try:
        config_json = json.loads(args.config)
    except json.JSONDecodeError:
        # print("Couldn't find a valid config file, using default arguments...")
        config_json = None

    if config_json is not None:
        for k, v in config_json:
            if vars(args)[k] == parser.get_default(k):
                vars(args)[k] = v

    return args


def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno is not errno.EEXIST:
            raise




