import logging

from django.conf import settings
from django.core.handlers.wsgi import WSGIRequest
from django.http import JsonResponse
from django.middleware.common import MiddlewareMixin



class ExceptionMiddleware(MiddlewareMixin):
    """统一异常处理中间件"""
    def process_exception(self, request:WSGIRequest, exception):
        """
        统一异常处理
        :param request: 请求对象
        :param exception: 异常对象
        :return:
        """
        # logger_error = logging.getLogger('django_ERROR')
        # request_body = request.body
        # request_url = request.path
        # request_method = request.method
        # error_msg = '\nRequest Body:{},\nRequest Url:{},\nRequest Method:{},\n{}'.format(request_body, request_url,
        #                                                                                  request_method,
        #                                                                                  exception)
        # logger_error.error(error_msg)
        # print(error_msg)

        if settings.DEBUG is True:
            raise exception

        return JsonResponse({
            'status': '500',
            'msg': str(exception)
        })