"""
For implementation details, refer to this source:
https://docs.microsoft.com/en-us/graph/api/resources/todo-overview?view=graph-rest-1.0
"""
import sys
sys.path.append("")
import json
from datetime import datetime
from typing import Union

from todocli.models.todolist import TodoList
from todocli.models.todotask import Task, TaskStatus
from todocli.graphapi.oauth import get_oauth_session

from todocli.utils.datetime_util import datetime_to_api_timestamp
import json
from requests_oauthlib import OAuth2Session
# Oauth settings
import os
import pickle
import time

import yaml

BASE_URL = "https://graph.microsoft.com/v1.0/me/todo/lists"


settings = {
    "redirect": "https://localhost/login/authorized",
    "scopes": "openid offline_access tasks.readwrite",
    "authority": "https://login.microsoftonline.com/common",
    "authorize_endpoint": "/oauth2/v2.0/authorize",
    "token_endpoint": "/oauth2/v2.0/token",
}

# Code taken from https://docs.microsoft.com/en-us/graph/tutorials/python?tutorial-step=3

# This is necessary because Azure does not guarantee
# to return scopes in the same case and order as requested
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
os.environ["OAUTHLIB_IGNORE_SCOPE_CHANGE"] = "1"

redirect = settings["redirect"]
scope = settings["scopes"]

authorize_url = "{0}{1}".format(settings["authority"], settings["authorize_endpoint"])
token_url = "{0}{1}".format(settings["authority"], settings["token_endpoint"])

# User settings location
config_dir = "{}/.config/tod0".format(os.path.expanduser("~"))
if not os.path.isdir(config_dir):
    os.makedirs(config_dir)


def check_keys(keys):
    client_id = keys["client_id"]
    client_secret = keys["client_secret"]

    if client_id == "" or client_secret == "":
        print(
            "Please create and enter your client id and secret in {}".format(
                os.path.join(config_dir, "keys.yml")
            )
        )
        print(
            "Instructions to getting your API client id and secret can be found here:\n{}".format(
                "https://github.com/kiblee/tod0/blob/master/GET_KEY.md"
            )
        )
        exit()


# Check for api keys
keys_path = os.path.join(config_dir, "keys.yml")
if not os.path.isfile(keys_path):
    keys = {"client_id": "", "client_secret": ""}

    with open(keys_path, "w") as f:
        yaml.dump(keys, f)
    check_keys(keys)
else:
    # Load api keys
    with open(keys_path) as f:
        keys = yaml.load(f, yaml.SafeLoader)
        check_keys(keys)

client_id = keys["client_id"]
client_secret = keys["client_secret"]


def get_token():
    try:
        # Try to load token from local
        with open(os.path.join(config_dir, "token.pkl"), "rb") as f:
            token = pickle.load(f)

        token = refresh_token(token)

    except Exception:
        # Authorize user to get token
        outlook = OAuth2Session(client_id, scope=scope, redirect_uri=redirect)

        # Redirect  the user owner to the OAuth provider
        authorization_url, state = outlook.authorization_url(authorize_url)
        print("Please go here and authorize:\n", authorization_url)

        # Get the authorization verifier code from the callback url
        redirect_response = input("Paste the full redirect URL below:\n")

        # Fetch the access token
        token = outlook.fetch_token(
            token_url,
            client_secret=client_secret,
            authorization_response=redirect_response,
        )

    store_token(token)
    return token


def store_token(token):
    with open(os.path.join(config_dir, "token.pkl"), "wb") as f:
        pickle.dump(token, f)


def refresh_token(token):
    # Check expiration
    now = time.time()
    # Subtract 5 minutes from expiration to account for clock skew
    expire_time = token["expires_at"] - 300
    if now >= expire_time:
        # Refresh the token
        aad_auth = OAuth2Session(
            client_id, token=token, scope=scope, redirect_uri=redirect
        )

        refresh_params = {"client_id": client_id, "client_secret": client_secret}

        new_token = aad_auth.refresh_token(token_url, **refresh_params)
        return new_token

    # Token still valid, just return it
    return token


def get_oauth_session():
    token = get_token()
    return OAuth2Session(client_id, scope=scope, token=token)

class ListNotFound(Exception):
    def __init__(self, list_name):
        self.message = "List with name '{}' could not be found".format(list_name)
        super(ListNotFound, self).__init__(self.message)


class TaskNotFoundByName(Exception):
    def __init__(self, task_name, list_name):
        self.message = "Task with name '{}' could not be found in list '{}'".format(
            task_name, list_name
        )
        super(TaskNotFoundByName, self).__init__(self.message)


class TaskNotFoundByIndex(Exception):
    def __init__(self, task_index, list_name):
        self.message = "Task with index '{}' could not be found in list '{}'".format(
            task_index, list_name
        )
        super(TaskNotFoundByIndex, self).__init__(self.message)


class MicrosoftToDo:
    def __init__(self) -> None:
        pass
        
    def parse_response(self,response):
        return json.loads(response.content.decode())["value"]


    def get_lists(self):
        session = get_oauth_session()
        response = session.get(BASE_URL)
        response_value = self.parse_response(response)
        list = [item['displayName'] for item in response_value]
        print('Lists: ', list)
        return response_value



    def create_list(self, title: str):
        request_body = {"displayName": title}
        session = get_oauth_session()
        response = session.post(BASE_URL, json=request_body)
        return True if response.ok else response.raise_for_status()


    # TODO No associated command
    def rename_list(self, old_title: str, new_title: str):
        list_id = self.get_list_id_by_name(old_title)
        request_body = {"title": new_title}
        session = get_oauth_session()
        response = session.patch(f"{BASE_URL}/{list_id}", json=request_body)
        return True if response.ok else response.raise_for_status()


    def get_tasks(self, list_name: str = None, list_id: str = None, num_tasks: int = 100):
        assert (list_name is not None) or (
            list_id is not None
        ), "You must provide list_name or list_id"

        # For compatibility with cli
        if list_id is None:
            list_id = self.get_list_id_by_name(list_name)

        endpoint = (
            f"{BASE_URL}/{list_id}/tasks?$filter=status ne 'completed'&$top={num_tasks}"
        )
        session = get_oauth_session()
        response = session.get(endpoint)
        response_value = self.parse_response(response)
        return response_value

    def get_short_form_tasks(self, list_name: str = None, list_id: str = None, num_tasks: int = 100):
        new_response = []
        for task in self.get_tasks(list_name, list_id, num_tasks):
            new_task = {
                'importance': task.get('importance', None),
                'isReminderOn': task.get('isReminderOn', None),
                'title': task.get('title', None),
                'createdDateTime': task.get('createdDateTime', None),
                'hasAttachments': task.get('hasAttachments', None),
                'checklistItems': task.get('checklistItems', None),
                'body': task.get('body', None),
            }
            new_response.append(new_task)
        return new_response


    def complete_task(self,
                    list_name: str = None,
                    task_name: Union[str, int] = None,
                    list_id: str = None,
                    task_id: str = None,
                ):
        assert (list_name is not None) or (
            list_id is not None
        ), "You must provide list_name or list_id"
        assert (task_name is not None) or (
            task_id is not None
        ), "You must provide task_name or task_id"

        # For compatibility with cli
        if list_id is None:
            list_id = self.get_list_id_by_name(list_name)
        if task_id is None:
            task_id = self.get_task_id_by_name(list_name, task_name)

        endpoint = f"{BASE_URL}/{list_id}/tasks/{task_id}"
        request_body = {
            "status": TaskStatus.COMPLETED,
            "completedDateTime": datetime_to_api_timestamp(datetime.now()),
        }
        session = get_oauth_session()
        response = session.patch(endpoint, json=request_body)
        return True if response.ok else response.raise_for_status()


    def remove_task(self, list_name: str, task_name: Union[str, int]):
        list_id = self.get_list_id_by_name(list_name)
        task_id = self.get_task_id_by_name(list_name, task_name)
        endpoint = f"{BASE_URL}/{list_id}/tasks/{task_id}"
        session = get_oauth_session()
        response = session.delete(endpoint)
        return True if response.ok else response.raise_for_status()


    def get_list_id_by_name(self, list_name):
        endpoint = f"{BASE_URL}?$filter=startswith(displayName,'{list_name}')"
        session = get_oauth_session()
        response = session.get(endpoint)
        response_value = self.parse_response(response)
        try:
            return response_value[0]["id"]
        except IndexError:
            raise ListNotFound(list_name)


    def get_task_id_by_name(self, list_name: str, task_name: str):
        if isinstance(task_name, str):
            try:
                list_id = self.get_list_id_by_name(list_name)
                endpoint = f"{BASE_URL}/{list_id}/tasks?$filter=title eq '{task_name}'"
                session = get_oauth_session()
                response = session.get(endpoint)
                response_value = self.parse_response(response)
                return [Task(x) for x in response_value][0].id
            except IndexError:
                raise TaskNotFoundByName(task_name, list_name)
        # elif isinstance(task_name, int):
        #    tasks = get_tasks(list_name, task_list_position + 1)
        #    try:
        #        return tasks[task_list_position].id
        #    except IndexError:
        #        raise TaskNotFoundByIndex(task_list_position, list_name)
        else:
            raise

    def create_task(self,
                        task_name: str,
                        list_name: str = None,
                        list_id: str = None,
                        importance:bool = False,
                        dueDateTime: datetime = None,
                        body=None,
                    ):
        assert (list_name is not None) or (
            list_id is not None
        ), "You must provide list_name or list_id"

        # For compatibility with cli
        if list_id is None:
            list_id = self.get_list_id_by_name(list_name)

        if body is None:
            body ={"content": task_name,
                    "contentType": "text"
                    }
        
        if dueDateTime is not None:
            dueDateTime = datetime.strptime(dueDateTime, '%Y-%m-%d:%H:%M:%S')
        
        if importance == False:
            importance = 'normal'
        else:
            importance = 'high'
        
        endpoint = f"{BASE_URL}/{list_id}/tasks"
        request_body = {
            "title": task_name,
            "body":body,
            "importance": importance,
            "dueDateTime": datetime_to_api_timestamp(dueDateTime),
        }
        session = get_oauth_session()
        response = session.post(endpoint, json=request_body)
        return True if response.ok else response.raise_for_status()