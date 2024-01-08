{"payload":{"allShortcutsEnabled":false,"fileTree":{"diffusion_policy/gym_util":{"items":[{"name":"async_vector_env.py","path":"diffusion_policy/gym_util/async_vector_env.py","contentType":"file"},{"name":"multistep_wrapper.py","path":"diffusion_policy/gym_util/multistep_wrapper.py","contentType":"file"},{"name":"sync_vector_env.py","path":"diffusion_policy/gym_util/sync_vector_env.py","contentType":"file"},{"name":"video_recording_wrapper.py","path":"diffusion_policy/gym_util/video_recording_wrapper.py","contentType":"file"},{"name":"video_wrapper.py","path":"diffusion_policy/gym_util/video_wrapper.py","contentType":"file"}],"totalCount":5},"diffusion_policy":{"items":[{"name":"codecs","path":"diffusion_policy/codecs","contentType":"directory"},{"name":"common","path":"diffusion_policy/common","contentType":"directory"},{"name":"config","path":"diffusion_policy/config","contentType":"directory"},{"name":"dataset","path":"diffusion_policy/dataset","contentType":"directory"},{"name":"env","path":"diffusion_policy/env","contentType":"directory"},{"name":"env_runner","path":"diffusion_policy/env_runner","contentType":"directory"},{"name":"gym_util","path":"diffusion_policy/gym_util","contentType":"directory"},{"name":"model","path":"diffusion_policy/model","contentType":"directory"},{"name":"policy","path":"diffusion_policy/policy","contentType":"directory"},{"name":"real_world","path":"diffusion_policy/real_world","contentType":"directory"},{"name":"scripts","path":"diffusion_policy/scripts","contentType":"directory"},{"name":"shared_memory","path":"diffusion_policy/shared_memory","contentType":"directory"},{"name":"workspace","path":"diffusion_policy/workspace","contentType":"directory"}],"totalCount":13},"":{"items":[{"name":"diffusion_policy","path":"diffusion_policy","contentType":"directory"},{"name":"media","path":"media","contentType":"directory"},{"name":"tests","path":"tests","contentType":"directory"},{"name":".gitignore","path":".gitignore","contentType":"file"},{"name":"LICENSE","path":"LICENSE","contentType":"file"},{"name":"README.md","path":"README.md","contentType":"file"},{"name":"conda_environment.yaml","path":"conda_environment.yaml","contentType":"file"},{"name":"conda_environment_macos.yaml","path":"conda_environment_macos.yaml","contentType":"file"},{"name":"conda_environment_real.yaml","path":"conda_environment_real.yaml","contentType":"file"},{"name":"demo_pusht.py","path":"demo_pusht.py","contentType":"file"},{"name":"demo_real_robot.py","path":"demo_real_robot.py","contentType":"file"},{"name":"eval.py","path":"eval.py","contentType":"file"},{"name":"eval_real_robot.py","path":"eval_real_robot.py","contentType":"file"},{"name":"multirun_metrics.py","path":"multirun_metrics.py","contentType":"file"},{"name":"pyrightconfig.json","path":"pyrightconfig.json","contentType":"file"},{"name":"ray_exec.py","path":"ray_exec.py","contentType":"file"},{"name":"ray_train_multirun.py","path":"ray_train_multirun.py","contentType":"file"},{"name":"setup.py","path":"setup.py","contentType":"file"},{"name":"train.py","path":"train.py","contentType":"file"}],"totalCount":19}},"fileTreeProcessingTime":10.294328,"foldersToFetch":[],"reducedMotionEnabled":null,"repo":{"id":610943527,"defaultBranch":"main","name":"diffusion_policy","ownerLogin":"real-stanford","currentUserCanPush":false,"isFork":false,"isEmpty":false,"createdAt":"2023-03-07T19:48:26.000Z","ownerAvatar":"https://avatars.githubusercontent.com/u/63878789?v=4","public":true,"private":false,"isOrgOwned":true},"symbolsExpanded":false,"treeExpanded":true,"refInfo":{"name":"main","listCacheKey":"v0:1696280712.0","canEdit":false,"refType":"branch","currentOid":"548a52bbb105518058e27bf34dcf90bf6f73681a"},"path":"diffusion_policy/gym_util/video_wrapper.py","currentUser":null,"blob":{"rawLines":["import gym","import numpy as np","","class VideoWrapper(gym.Wrapper):","    def __init__(self, ","            env, ","            mode='rgb_array',","            enabled=True,","            steps_per_render=1,","            **kwargs","        ):","        super().__init__(env)","        ","        self.mode = mode","        self.enabled = enabled","        self.render_kwargs = kwargs","        self.steps_per_render = steps_per_render","","        self.frames = list()","        self.step_count = 0","","    def reset(self, **kwargs):","        obs = super().reset(**kwargs)","        self.frames = list()","        self.step_count = 1","        if self.enabled:","            frame = self.env.render(","                mode=self.mode, **self.render_kwargs)","            assert frame.dtype == np.uint8","            self.frames.append(frame)","        return obs","    ","    def step(self, action):","        result = super().step(action)","        self.step_count += 1","        if self.enabled and ((self.step_count % self.steps_per_render) == 0):","            frame = self.env.render(","                mode=self.mode, **self.render_kwargs)","            assert frame.dtype == np.uint8","            self.frames.append(frame)","        return result","    ","    def render(self, mode='rgb_array', **kwargs):","        return self.frames"],"stylingDirectives":[[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":10,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":15,"cssClass":"pl-k"},{"start":16,"end":18,"cssClass":"pl-s1"}],[],[{"start":0,"end":5,"cssClass":"pl-k"},{"start":6,"end":18,"cssClass":"pl-v"},{"start":19,"end":22,"cssClass":"pl-s1"},{"start":23,"end":30,"cssClass":"pl-v"}],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":21,"cssClass":"pl-s1"}],[{"start":12,"end":15,"cssClass":"pl-s1"}],[{"start":12,"end":16,"cssClass":"pl-s1"},{"start":16,"end":17,"cssClass":"pl-c1"},{"start":17,"end":28,"cssClass":"pl-s"}],[{"start":12,"end":19,"cssClass":"pl-s1"},{"start":19,"end":20,"cssClass":"pl-c1"},{"start":20,"end":24,"cssClass":"pl-c1"}],[{"start":12,"end":28,"cssClass":"pl-s1"},{"start":28,"end":29,"cssClass":"pl-c1"},{"start":29,"end":30,"cssClass":"pl-c1"}],[{"start":12,"end":14,"cssClass":"pl-c1"},{"start":14,"end":20,"cssClass":"pl-s1"}],[],[{"start":8,"end":13,"cssClass":"pl-en"},{"start":16,"end":24,"cssClass":"pl-en"},{"start":25,"end":28,"cssClass":"pl-s1"}],[],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":20,"end":24,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":20,"cssClass":"pl-s1"},{"start":21,"end":22,"cssClass":"pl-c1"},{"start":23,"end":30,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":26,"cssClass":"pl-s1"},{"start":27,"end":28,"cssClass":"pl-c1"},{"start":29,"end":35,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":29,"cssClass":"pl-s1"},{"start":30,"end":31,"cssClass":"pl-c1"},{"start":32,"end":48,"cssClass":"pl-s1"}],[],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":21,"cssClass":"pl-c1"},{"start":22,"end":26,"cssClass":"pl-en"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":23,"cssClass":"pl-s1"},{"start":24,"end":25,"cssClass":"pl-c1"},{"start":26,"end":27,"cssClass":"pl-c1"}],[],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":13,"cssClass":"pl-en"},{"start":14,"end":18,"cssClass":"pl-s1"},{"start":20,"end":22,"cssClass":"pl-c1"},{"start":22,"end":28,"cssClass":"pl-s1"}],[{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":14,"end":19,"cssClass":"pl-en"},{"start":22,"end":27,"cssClass":"pl-en"},{"start":28,"end":30,"cssClass":"pl-c1"},{"start":30,"end":36,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":21,"cssClass":"pl-c1"},{"start":22,"end":26,"cssClass":"pl-en"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":23,"cssClass":"pl-s1"},{"start":24,"end":25,"cssClass":"pl-c1"},{"start":26,"end":27,"cssClass":"pl-c1"}],[{"start":8,"end":10,"cssClass":"pl-k"},{"start":11,"end":15,"cssClass":"pl-s1"},{"start":16,"end":23,"cssClass":"pl-s1"}],[{"start":12,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":20,"end":24,"cssClass":"pl-s1"},{"start":25,"end":28,"cssClass":"pl-s1"},{"start":29,"end":35,"cssClass":"pl-en"}],[{"start":16,"end":20,"cssClass":"pl-s1"},{"start":20,"end":21,"cssClass":"pl-c1"},{"start":21,"end":25,"cssClass":"pl-s1"},{"start":26,"end":30,"cssClass":"pl-s1"},{"start":32,"end":34,"cssClass":"pl-c1"},{"start":34,"end":38,"cssClass":"pl-s1"},{"start":39,"end":52,"cssClass":"pl-s1"}],[{"start":12,"end":18,"cssClass":"pl-k"},{"start":19,"end":24,"cssClass":"pl-s1"},{"start":25,"end":30,"cssClass":"pl-s1"},{"start":31,"end":33,"cssClass":"pl-c1"},{"start":34,"end":36,"cssClass":"pl-s1"},{"start":37,"end":42,"cssClass":"pl-s1"}],[{"start":12,"end":16,"cssClass":"pl-s1"},{"start":17,"end":23,"cssClass":"pl-s1"},{"start":24,"end":30,"cssClass":"pl-en"},{"start":31,"end":36,"cssClass":"pl-s1"}],[{"start":8,"end":14,"cssClass":"pl-k"},{"start":15,"end":18,"cssClass":"pl-s1"}],[],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":12,"cssClass":"pl-en"},{"start":13,"end":17,"cssClass":"pl-s1"},{"start":19,"end":25,"cssClass":"pl-s1"}],[{"start":8,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":17,"end":22,"cssClass":"pl-en"},{"start":25,"end":29,"cssClass":"pl-en"},{"start":30,"end":36,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":23,"cssClass":"pl-s1"},{"start":24,"end":26,"cssClass":"pl-c1"},{"start":27,"end":28,"cssClass":"pl-c1"}],[{"start":8,"end":10,"cssClass":"pl-k"},{"start":11,"end":15,"cssClass":"pl-s1"},{"start":16,"end":23,"cssClass":"pl-s1"},{"start":24,"end":27,"cssClass":"pl-c1"},{"start":30,"end":34,"cssClass":"pl-s1"},{"start":35,"end":45,"cssClass":"pl-s1"},{"start":46,"end":47,"cssClass":"pl-c1"},{"start":48,"end":52,"cssClass":"pl-s1"},{"start":53,"end":69,"cssClass":"pl-s1"},{"start":71,"end":73,"cssClass":"pl-c1"},{"start":74,"end":75,"cssClass":"pl-c1"}],[{"start":12,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":20,"end":24,"cssClass":"pl-s1"},{"start":25,"end":28,"cssClass":"pl-s1"},{"start":29,"end":35,"cssClass":"pl-en"}],[{"start":16,"end":20,"cssClass":"pl-s1"},{"start":20,"end":21,"cssClass":"pl-c1"},{"start":21,"end":25,"cssClass":"pl-s1"},{"start":26,"end":30,"cssClass":"pl-s1"},{"start":32,"end":34,"cssClass":"pl-c1"},{"start":34,"end":38,"cssClass":"pl-s1"},{"start":39,"end":52,"cssClass":"pl-s1"}],[{"start":12,"end":18,"cssClass":"pl-k"},{"start":19,"end":24,"cssClass":"pl-s1"},{"start":25,"end":30,"cssClass":"pl-s1"},{"start":31,"end":33,"cssClass":"pl-c1"},{"start":34,"end":36,"cssClass":"pl-s1"},{"start":37,"end":42,"cssClass":"pl-s1"}],[{"start":12,"end":16,"cssClass":"pl-s1"},{"start":17,"end":23,"cssClass":"pl-s1"},{"start":24,"end":30,"cssClass":"pl-en"},{"start":31,"end":36,"cssClass":"pl-s1"}],[{"start":8,"end":14,"cssClass":"pl-k"},{"start":15,"end":21,"cssClass":"pl-s1"}],[],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":14,"cssClass":"pl-en"},{"start":15,"end":19,"cssClass":"pl-s1"},{"start":21,"end":25,"cssClass":"pl-s1"},{"start":25,"end":26,"cssClass":"pl-c1"},{"start":26,"end":37,"cssClass":"pl-s"},{"start":39,"end":41,"cssClass":"pl-c1"},{"start":41,"end":47,"cssClass":"pl-s1"}],[{"start":8,"end":14,"cssClass":"pl-k"},{"start":15,"end":19,"cssClass":"pl-s1"},{"start":20,"end":26,"cssClass":"pl-s1"}]],"csv":null,"csvError":null,"dependabotInfo":{"showConfigurationBanner":false,"configFilePath":null,"networkDependabotPath":"/real-stanford/diffusion_policy/network/updates","dismissConfigurationNoticePath":"/settings/dismiss-notice/dependabot_configuration_notice","configurationNoticeDismissed":null,"repoAlertsPath":"/real-stanford/diffusion_policy/security/dependabot","repoSecurityAndAnalysisPath":"/real-stanford/diffusion_policy/settings/security_analysis","repoOwnerIsOrg":true,"currentUserCanAdminRepo":false},"displayName":"video_wrapper.py","displayUrl":"https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/gym_util/video_wrapper.py?raw=true","headerInfo":{"blobSize":"1.23 KB","deleteInfo":{"deleteTooltip":"You must be signed in to make or propose changes"},"editInfo":{"editTooltip":"You must be signed in to make or propose changes"},"ghDesktopPath":"https://desktop.github.com","gitLfsPath":null,"onBranch":true,"shortPath":"abfebbe","siteNavLoginPath":"/login?return_to=https%3A%2F%2Fgithub.com%2Freal-stanford%2Fdiffusion_policy%2Fblob%2Fmain%2Fdiffusion_policy%2Fgym_util%2Fvideo_wrapper.py","isCSV":false,"isRichtext":false,"toc":null,"lineInfo":{"truncatedLoc":"44","truncatedSloc":"38"},"mode":"file"},"image":false,"isCodeownersFile":null,"isPlain":false,"isValidLegacyIssueTemplate":false,"issueTemplateHelpUrl":"https://docs.github.com/articles/about-issue-and-pull-request-templates","issueTemplate":null,"discussionTemplate":null,"language":"Python","languageID":303,"large":false,"loggedIn":false,"newDiscussionPath":"/real-stanford/diffusion_policy/discussions/new","newIssuePath":"/real-stanford/diffusion_policy/issues/new","planSupportInfo":{"repoIsFork":null,"repoOwnedByCurrentUser":null,"requestFullPath":"/real-stanford/diffusion_policy/blob/main/diffusion_policy/gym_util/video_wrapper.py","showFreeOrgGatedFeatureMessage":null,"showPlanSupportBanner":null,"upgradeDataAttributes":null,"upgradePath":null},"publishBannersInfo":{"dismissActionNoticePath":"/settings/dismiss-notice/publish_action_from_dockerfile","dismissStackNoticePath":"/settings/dismiss-notice/publish_stack_from_file","releasePath":"/real-stanford/diffusion_policy/releases/new?marketplace=true","showPublishActionBanner":false,"showPublishStackBanner":false},"rawBlobUrl":"https://github.com/real-stanford/diffusion_policy/raw/main/diffusion_policy/gym_util/video_wrapper.py","renderImageOrRaw":false,"richText":null,"renderedFileInfo":null,"shortPath":null,"tabSize":8,"topBannersInfo":{"overridingGlobalFundingFile":false,"globalPreferredFundingPath":null,"repoOwner":"real-stanford","repoName":"diffusion_policy","showInvalidCitationWarning":false,"citationHelpUrl":"https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files","showDependabotConfigurationBanner":false,"actionsOnboardingTip":null},"truncated":false,"viewable":true,"workflowRedirectUrl":null,"symbols":{"timed_out":false,"not_analyzed":false,"symbols":[{"name":"VideoWrapper","kind":"class","ident_start":37,"ident_end":49,"extent_start":31,"extent_end":1260,"fully_qualified_name":"VideoWrapper","ident_utf16":{"start":{"line_number":3,"utf16_col":6},"end":{"line_number":3,"utf16_col":18}},"extent_utf16":{"start":{"line_number":3,"utf16_col":0},"end":{"line_number":43,"utf16_col":26}}},{"name":"__init__","kind":"function","ident_start":72,"ident_end":80,"extent_start":68,"extent_end":463,"fully_qualified_name":"VideoWrapper.__init__","ident_utf16":{"start":{"line_number":4,"utf16_col":8},"end":{"line_number":4,"utf16_col":16}},"extent_utf16":{"start":{"line_number":4,"utf16_col":4},"end":{"line_number":19,"utf16_col":27}}},{"name":"reset","kind":"function","ident_start":473,"ident_end":478,"extent_start":469,"extent_end":806,"fully_qualified_name":"VideoWrapper.reset","ident_utf16":{"start":{"line_number":21,"utf16_col":8},"end":{"line_number":21,"utf16_col":13}},"extent_utf16":{"start":{"line_number":21,"utf16_col":4},"end":{"line_number":30,"utf16_col":18}}},{"name":"step","kind":"function","ident_start":820,"ident_end":824,"extent_start":816,"extent_end":1178,"fully_qualified_name":"VideoWrapper.step","ident_utf16":{"start":{"line_number":32,"utf16_col":8},"end":{"line_number":32,"utf16_col":12}},"extent_utf16":{"start":{"line_number":32,"utf16_col":4},"end":{"line_number":40,"utf16_col":21}}},{"name":"render","kind":"function","ident_start":1192,"ident_end":1198,"extent_start":1188,"extent_end":1260,"fully_qualified_name":"VideoWrapper.render","ident_utf16":{"start":{"line_number":42,"utf16_col":8},"end":{"line_number":42,"utf16_col":14}},"extent_utf16":{"start":{"line_number":42,"utf16_col":4},"end":{"line_number":43,"utf16_col":26}}}]}},"copilotInfo":null,"copilotAccessAllowed":false,"csrf_tokens":{"/real-stanford/diffusion_policy/branches":{"post":"8WAf2CDkwwSZSWVsaj8wxmNgpioLgbi1tLoOXAtn3D8DNFfASfgXUZR20fQGccDo0E4-VRs2IH2c31msVv_XfQ"},"/repos/preferences":{"post":"uswPUum4El1zxqcHMscLkiX6IOh3edzepnPgknJBe6c8sjDyi3lvXUjSUJMBHDryyE3TIVmo0mY7Ds8hO4Q0Pg"}}},"title":"diffusion_policy/diffusion_policy/gym_util/video_wrapper.py at main · real-stanford/diffusion_policy"}