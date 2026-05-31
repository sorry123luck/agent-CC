"""
动作配方管理器

负责保存、加载动作执行配方。
"""
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from src.storage.db import Session as DBSession
from src.storage.schema import ActionRecipe


@dataclass
class ActionStep:
    """动作步骤"""
    action: str  # click / send_text / invoke_uia
    target: str  # 元素键或坐标描述
    value: str | None = None  # 文本输入时的值


@dataclass
class ActionRecipeData:
    """动作配方"""
    app_id: str
    page_type: str
    action_name: str
    steps: list[ActionStep]
    success_rule: dict | None = None
    fallback_rule: dict | None = None


class ActionRecipeManager:
    """
    管理动作配方的保存、加载和执行

    配方结构：
    - app_id + page_type：标识配方适用的页面
    - action_name：动作名称（如 "打开文件"）
    - steps：执行步骤列表
    - success_rule：成功判定规则
    - fallback_rule：失败备用规则
    """

    def save(self, recipe: ActionRecipeData) -> ActionRecipe:
        """保存动作配方"""
        with DBSession() as s:
            existing = s.query(ActionRecipe).filter(
                ActionRecipe.app_id == recipe.app_id,
                ActionRecipe.page_type == recipe.page_type,
                ActionRecipe.action_name == recipe.action_name,
            ).first()

            steps_json = json.dumps([
                {"action": step.action, "target": step.target, "value": step.value}
                for step in recipe.steps
            ], ensure_ascii=False)

            if existing:
                existing.steps_json = steps_json
                existing.success_rule_json = json.dumps(recipe.success_rule) if recipe.success_rule else None
                existing.fallback_rule_json = json.dumps(recipe.fallback_rule) if recipe.fallback_rule else None
                existing.updated_at = datetime.now(timezone.utc)
                s.commit()
                s.refresh(existing)
                return existing

            new_recipe = ActionRecipe(
                app_id=recipe.app_id,
                page_type=recipe.page_type,
                action_name=recipe.action_name,
                steps_json=steps_json,
                success_rule_json=json.dumps(recipe.success_rule) if recipe.success_rule else None,
                fallback_rule_json=json.dumps(recipe.fallback_rule) if recipe.fallback_rule else None,
            )
            s.add(new_recipe)
            s.commit()
            s.refresh(new_recipe)
            return new_recipe

    def load(
        self,
        app_id: str,
        page_type: str,
        action_name: str,
    ) -> ActionRecipeData | None:
        """加载动作配方"""
        with DBSession() as s:
            recipe = s.query(ActionRecipe).filter(
                ActionRecipe.app_id == app_id,
                ActionRecipe.page_type == page_type,
                ActionRecipe.action_name == action_name,
            ).first()

            if not recipe:
                return None

            steps = []
            if recipe.steps_json:
                for step_data in json.loads(recipe.steps_json):
                    steps.append(ActionStep(
                        action=step_data["action"],
                        target=step_data["target"],
                        value=step_data.get("value"),
                    ))

            return ActionRecipeData(
                app_id=recipe.app_id,
                page_type=recipe.page_type,
                action_name=recipe.action_name,
                steps=steps,
                success_rule=json.loads(recipe.success_rule_json) if recipe.success_rule_json else None,
                fallback_rule=json.loads(recipe.fallback_rule_json) if recipe.fallback_rule_json else None,
            )

    def list_recipes(self, app_id: str, page_type: str | None = None) -> list[ActionRecipeData]:
        """列出动作配方"""
        with DBSession() as s:
            query = s.query(ActionRecipe).filter(ActionRecipe.app_id == app_id)
            if page_type:
                query = query.filter(ActionRecipe.page_type == page_type)

            results = []
            for recipe in query.all():
                steps = []
                if recipe.steps_json:
                    for step_data in json.loads(recipe.steps_json):
                        steps.append(ActionStep(
                            action=step_data["action"],
                            target=step_data["target"],
                            value=step_data.get("value"),
                        ))
                results.append(ActionRecipeData(
                    app_id=recipe.app_id,
                    page_type=recipe.page_type,
                    action_name=recipe.action_name,
                    steps=steps,
                    success_rule=json.loads(recipe.success_rule_json) if recipe.success_rule_json else None,
                    fallback_rule=json.loads(recipe.fallback_rule_json) if recipe.fallback_rule_json else None,
                ))
            return results
