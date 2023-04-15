from datetime import datetime
import uuid
from typing import Optional

from logging import getLogger
import pytz
from sqlalchemy import Column, Boolean, Text, ForeignKey, DateTime, Integer, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import registry, declarative_base
from sqlalchemy_mixins.activerecord import ActiveRecordMixin
from sqlalchemy_mixins.inspection import InspectionMixin
from sqlalchemy_mixins.smartquery import SmartQueryMixin

mapper_registry = registry()
DeclarativeBase = declarative_base()
Base = mapper_registry.generate_base(
    cls=(DeclarativeBase, ActiveRecordMixin, SmartQueryMixin, InspectionMixin)
)

logger = getLogger(__name__)

@mapper_registry.mapped
class ModelMixin(Base):
    """
    Generic Mixin Model to provide CRUD functions that all Model classes need
    """

    __abstract__ = True

    @classmethod
    def create(cls, *_, **kwargs) -> 'Base':
        """
        This function creates a new instance of a class, adds it to a session, commits the changes, and returns the instance.

        :param cls: The `cls` parameter refers to the class that the method is defined in.
        In this case, it is a class method defined in a class that inherits from `Base`
        :return: an instance of the class `Base` that was created with the provided `kwargs` arguments.
        """
        data = cls(**kwargs)
        cls.session.add(data)
        cls.session.commit()
        return data

    @classmethod
    def update(cls, *_, **kwargs) -> 'Base':
        """
        This function updates an object in a database and returns the updated object.

        :param cls: The `cls` parameter is a reference to the class itself.
        It is used to call class methods and access class attributes. In this case,
        it is used to access the database session and query the database for objects of the class
        :return: The method is returning the updated object with the specified `id`.
        """
        object_id = kwargs.pop("id")
        _object = cls.where(id=object_id)
        _object.update(kwargs)
        cls.session.commit()
        return _object.first()

    @classmethod
    def remove(cls, *_, **kwargs) -> int:
        """
        This function removes an object from a class based on its ID.

        :param cls: The `cls` parameter is a reference to the class itself.
        It is used to access class-level methods and attributes. In this case, it is used to access the `where`
        method and delete the object with the specified `id`
        :return: the `object_id` which is a UUID (Universally Unique Identifier) of the deleted object.
        """
        object_id = kwargs.pop("id")
        try:
            cls.where(id=object_id).delete()
        except Exception as e:
            logger.info(e)
            cls.session.rollback()
        return object_id

    @classmethod
    def read(cls, *_, **kwargs) -> 'Base':
        """
        This function reads and returns the first instance of a class that
        matches the given keyword arguments.

        :param cls: The `cls` parameter is a reference to the class that the
        method is defined in. It is used to create an instance of the class and return it
        :return: The `read` method returns an instance of the `Base` class that
        matches the specified keyword arguments. It uses the `where` method to
        filter the instances and the
        `first` method to return the first instance that matches the filter.
        """
        return cls.where(**kwargs).first()

    @classmethod
    def read_all(cls) -> 'Base':
        """
        The function returns all instances of a class by querying the database.

        :param cls: The parameter `cls` refers to the class itself, which is
        passed as an argument to the method. This is a class method, which means it can be called on the class
        itself rather than an instance of the class. The method returns all
        instances of the class by calling the `where()` method
        :return: The method `read_all` is returning all instances of the `Base`
        class by calling the `where()` method with no arguments (which returns all instances) and then calling
        the `all()` method to retrieve all the instances.
        """
        return cls.where().all()


class History(ModelMixin):
    """
    This class represents the `history` table in the database.
    """
    __tablename__ = "history"
    id = Column(Integer, autoincrement=True, primary_key=True, default=uuid.uuid4, index=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=lambda: datetime.now(pytz.utc))
    spy_price = Column(Float, nullable=False)
    vix = Column(Float, nullable=False)
    expiration = Column(DateTime, nullable=False)
    pcr = Column(Float, nullable=False)
    put_notional = Column(Float, nullable=False)
    call_notional = Column(Float, nullable=False)


