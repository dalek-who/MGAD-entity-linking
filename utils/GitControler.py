from git import Repo, InvalidGitRepositoryError
from pathlib import Path
from functools import lru_cache

"""
利用服务器远程的git裸仓库辅助实验记录
"""


class GitController(object):
    def __init__(self, git_dir: Path=None):
        # 创建repo
        try:
            if git_dir is None:
                repo = Repo(search_parent_directories=True)
            else:
                repo = Repo(git_dir)  # bare
        except InvalidGitRepositoryError:
            repo = None
        self.repo = repo

    @lru_cache(maxsize=1)  # 只获取一次latest_commit就缓存起来，防止后面跑新的实验导致git变了
    def latest_commit(self):
        """
        当前最新的commit信息
        :return:
        """
        if self.repo is not None:
            sha = self.repo.head.object.hexsha
            short_sha = self.repo.git.rev_parse(sha, short=8)
            return {
                "git_sha": sha,
                "git_short_sha": short_sha,
                "git_time": str(self.repo.head.object.committed_datetime),
                "git_message": "\n" + self.repo.head.object.message,
            }
        else:
            return {
                "git_sha": "xxxx",
                "git_short_sha": "xxxx",
                "git_time": "",
                "git_message": "******  This is a fake message  ******",
            }

    # def assert_check(self):
    #     """
    #     repo状态检查
    #     :return:
    #     """
    #     assert not self.repo.bare, f"bare: {self.repo.bare}"  # 是否是裸(bare)仓库
    #     assert not self.repo.is_dirty(), f"dirty: {self.repo.is_dirty()}"  # 是否有add而未commit的脏文件
    #     assert not self.repo.untracked_files, f"untracked: {self.repo.untracked_files}"  # 是否有untracked文件

if __name__=="__main__":
    controller = GitController()
    latest = controller.latest_commit()
    latest_2 = controller.latest_commit()
    pass