from django.core.urlresolvers import reverse_lazy
from django.conf import settings
from django.contrib import messages, auth
from django.contrib.auth.decorators import login_required
from django.http import HttpResponsePermanentRedirect
from django.shortcuts import get_object_or_404, redirect
from django.utils.translation import ugettext_lazy as _
from django.template.response import TemplateResponse



def store_details(request, slug, store_id):
    pass


@login_required
def create(request):
    pass

